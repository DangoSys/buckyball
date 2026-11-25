package examples.designs.magia.devices.magia

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, Instantiate}
import framework.device.peripherals.Controller
import framework.device.peripherals.bootrom.BootRomLines
import framework.hart.memory.home.Home
import framework.generic.ddr.Bridge
import framework.generic.bus.chi.snf.LineResponse
import framework.generic.arbiter.RRArbiter
import framework.generic.queue.Queue
import framework.generic.bus.axi4
import framework.generic.bus.axi4.Decoder
import framework.hart.cbus.CBus
import framework.hart.cbus.generated.HartComponents
import buckyball.config.Design
import framework.device.sims.configs.GlobalConfig
import framework.device.dlink.BaseDevice

@instantiable
class DeviceTop(b: GlobalConfig, design: Design) extends BaseDevice(b.device.ddr.axi) {
  val deviceParam = b.device

  val Seq(hartParam) = b.harts

  val bus  =
    Instantiate(new CBus(b.harts, b.device.ddr, CBus.generators(design, b.harts, b.sim, HartComponents.factories)))
  val home = Instantiate(new Home(hartParam.memory.home, false, bus.pages.size))
  home.io.active            := true.B
  home.io.blockRequesterRsp := false.B
  home.io.requesters <> bus.coherent
  home.page <> bus.pages

  val peripherals =
    Instantiate(new Controller(deviceParam, deviceParam.ddr.axi, hartParam.memory.memoryAccess.memory))

  val peripheralBus = Instantiate(new Decoder(
    Seq(
      (deviceParam.clint.base, deviceParam.clint.bytes),
      (deviceParam.plic.base, deviceParam.plic.bytes),
      (deviceParam.scu.baseAddress, deviceParam.scu.totalSizeBytes),
      (deviceParam.trace.base, deviceParam.trace.bytes)
    ),
    deviceParam.ddr.axi
  ))

  peripheralBus.io.in <> bus.uncached
  peripherals.io.axi <> peripheralBus.io.matched
  bus.external <> peripheralBus.io.other
  peripherals.io.hardwareInterrupt.sources := 0.U
  bus.interrupts                           := peripherals.io.hardwareInterrupt.harts

  val rom     = Instantiate(new BootRomLines(deviceParam.bootrom, deviceParam.ddr.line))
  val bridge  = Instantiate(new Bridge(deviceParam.ddr))
  val request = home.io.memoryReq
  val inRom   = request.bits.addr >= deviceParam.bootrom.base.U &&
    request.bits.addr < (deviceParam.bootrom.base + deviceParam.bootrom.bytes).U
  rom.io.request.valid       := request.valid && inRom
  rom.io.request.bits        := request.bits
  bridge.io.request(0).valid := request.valid && !inRom
  bridge.io.request(0).bits  := request.bits
  request.ready              := Mux(inRom, rom.io.request.ready, bridge.io.request(0).ready)
  val responses = Instantiate(new RRArbiter(new LineResponse(deviceParam.ddr.line), 2))
  responses.io.in(0) <> rom.io.response
  responses.io.in(1) <> bridge.io.response(0)
  val response  = Instantiate(new Queue(new LineResponse(deviceParam.ddr.line), 1, false, false))
  response.io.enq <> responses.io.out
  home.io.memoryResp <> response.io.deq

  val memoryBus = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(deviceParam.ddr.axi, 2, None)))
  memoryBus.io.in(0) <> bus.dlink.axi
  memoryBus.io.in(1) <> bridge.io.axi
  dlink.axi <> memoryBus.io.out

}
