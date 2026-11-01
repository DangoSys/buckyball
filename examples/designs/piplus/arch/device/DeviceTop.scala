package examples.designs.piplus

import framework.lib.arbiter.RRArbiter

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.device.cbus.CBus
import framework.device.cbus.generated.CoreComponents
import buckyball.config.Design
import _root_.configs.GlobalConfig
import framework.core.memory.home.Home
import framework.device.ddr.Bridge
import framework.device.peripherals.Controller
import framework.device.peripherals.bootrom.BootRomLines
import framework.device.dlink.BaseDevice
import framework.ip.bus.axi4
import framework.ip.bus.chi.snf.LineResponse
import framework.lib.queue.{Queue => ResponseQueue}

/** The configured cores share one coherent Home in front of the boot ROM and DDR. */
@instantiable
class DeviceTop(b: GlobalConfig, design: Design) extends BaseDevice(b.device.ddr.axi) {
  val device     = b.device
  val cores      = device.cores
  val sharedHome = cores.head.memory.home
  require(cores.forall(_.memory.home == sharedHome), "Cores must share one Home")
  require(sharedHome.memory.agents == cores.map(_.clink.requesters).sum, "Home agents must count every requester")

  val bus: Instance[CBus] =
    Instantiate(new CBus(device, CBus.generators(design, device, b.sim, CoreComponents.factories)))

  val home:        Instance[Home]       = Instantiate(new Home(sharedHome))
  val peripherals: Instance[Controller] =
    Instantiate(new Controller(device, device.ddr.axi, cores.head.memory.memoryAccess.memory))

  val peripheralBus: Instance[axi4.Decoder] = Instantiate(new axi4.Decoder(
    Seq(
      (device.clint.base, device.clint.bytes),
      (device.plic.base, device.plic.bytes),
      (device.scu.baseAddress, device.scu.totalSizeBytes),
      (device.trace.base, device.trace.bytes)
    ),
    device.ddr.axi
  ))

  val rom:       Instance[BootRomLines]                = Instantiate(new BootRomLines(device.bootrom, sharedHome.memory.chi))
  val bridge:    Instance[Bridge]                      = Instantiate(new Bridge(device.ddr))
  val responses: Instance[RRArbiter[LineResponse]]     =
    Instantiate(new RRArbiter(new LineResponse(sharedHome.memory.chi), 2))
  val response:  Instance[ResponseQueue[LineResponse]] =
    Instantiate(new ResponseQueue(new LineResponse(sharedHome.memory.chi), 1, false, false))
  val memoryBus: Instance[axi4.Interconnect]           =
    Instantiate(new axi4.Interconnect(axi4.InterconnectParam(device.ddr.axi, 2, None)))

  home.io.active            := true.B
  home.io.blockRequesterRsp := false.B
  home.io.requesters <> bus.coherent

  peripheralBus.io.in <> bus.uncached
  peripherals.io.axi <> peripheralBus.io.matched
  bus.external <> peripheralBus.io.other
  peripherals.io.hardwareInterrupt.sources := 0.U
  bus.interrupts                           := peripherals.io.hardwareInterrupt.harts

  require(device.ddr.clients == 1)
  val request = home.io.memoryReq
  val inRom   = request.bits.addr >= device.bootrom.base.U &&
    request.bits.addr < (device.bootrom.base + device.bootrom.bytes).U
  rom.io.request.valid       := request.valid && inRom
  rom.io.request.bits        := request.bits
  bridge.io.request(0).valid := request.valid && !inRom
  bridge.io.request(0).bits  := request.bits
  request.ready              := Mux(inRom, rom.io.request.ready, bridge.io.request(0).ready)
  responses.io.in(0) <> rom.io.response
  responses.io.in(1) <> bridge.io.response(0)
  response.io.enq <> responses.io.out
  home.io.memoryResp <> response.io.deq

  memoryBus.io.in(0) <> bus.dlink.axi
  memoryBus.io.in(1) <> bridge.io.axi
  dlink.axi <> memoryBus.io.out
}
