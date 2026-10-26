package examples.designs.goban

import framework.lib.arbiter.{RRArbiter}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.device.configs.DeviceParam
import framework.device.cbus.CBus
import framework.device.cbus.generated.CoreComponents
import buckyball.config.Design
import _root_.configs.GlobalConfig
import framework.core.clink.CLinkIO
import framework.core.memory.home.Home
import framework.device.ddr.Bridge
import framework.device.peripherals.Controller
import framework.device.peripherals.bootrom.BootRomLines
import framework.device.dlink.BaseDevice
import framework.ip.bus.axi4
import framework.ip.bus.chi.snf.LineResponse
import framework.lib.queue.{Queue => ResponseQueue}

/** One Rocket hart schedules local MatMul threads through its core. */
@instantiable
class DeviceTop(b: GlobalConfig, design: Design) extends BaseDevice(b.device.ddr.axi) {
  val device = b.device

  val Seq(corep) = device.cores
  val sharedHome = corep.memory.home
  require(sharedHome.memory.agents == 3)
  val bus        = Instantiate(new CBus(b.device, CBus.generators(design, b.device, b.sim, CoreComponents.factories)))

  val home: Instance[Home] = Instantiate(new Home(sharedHome))
  home.io.active            := true.B
  home.io.blockRequesterRsp := false.B
  home.io.requesters <> bus.coherent

  val peripherals = Instantiate(new Controller(device, device.ddr.axi, corep.memory.memoryAccess.memory))

  val peripheralBus = Instantiate(new axi4.Decoder(
    Seq(
      (device.clint.base, device.clint.bytes),
      (device.plic.base, device.plic.bytes),
      (device.scu.baseAddress, device.scu.totalSizeBytes)
    ),
    device.ddr.axi
  ))

  peripheralBus.io.in <> bus.uncached
  peripherals.io.axi <> peripheralBus.io.matched
  bus.external <> peripheralBus.io.other
  peripherals.io.hardwareInterrupt.sources := 0.U
  bus.interrupts                           := peripherals.io.hardwareInterrupt.harts

  val rom     = Instantiate(new BootRomLines(device.bootrom, sharedHome.memory.chi))
  val bridge  = Instantiate(new Bridge(device.ddr))
  require(device.ddr.clients == 1)
  val request = home.io.memoryReq
  val inRom   = request.bits.addr >= device.bootrom.base.U &&
    request.bits.addr < (device.bootrom.base + device.bootrom.bytes).U
  rom.io.request.valid       := request.valid && inRom
  rom.io.request.bits        := request.bits
  bridge.io.request(0).valid := request.valid && !inRom
  bridge.io.request(0).bits  := request.bits
  request.ready              := Mux(inRom, rom.io.request.ready, bridge.io.request(0).ready)
  val responses = Instantiate(new RRArbiter(new LineResponse(sharedHome.memory.chi), 2))
  responses.io.in(0) <> rom.io.response
  responses.io.in(1) <> bridge.io.response(0)
  val response  = Instantiate(new ResponseQueue(new LineResponse(sharedHome.memory.chi), 1, false, false))
  response.io.enq <> responses.io.out
  home.io.memoryResp <> response.io.deq

  val memoryBus = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(device.ddr.axi, 2, None)))
  memoryBus.io.in(0) <> bus.dlink.axi
  memoryBus.io.in(1) <> bridge.io.axi
  dlink.axi <> memoryBus.io.out
}
