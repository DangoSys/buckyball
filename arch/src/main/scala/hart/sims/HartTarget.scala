package framework.hart.sims

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.device.sims.configs.GlobalConfig
import framework.hart.hlink.HLinkIO
import framework.device.dlink.BaseDevice
import framework.generic.ddr.Bridge
import framework.device.peripherals.Controller
import framework.device.peripherals.bootrom.BootRomLines
import framework.generic.bus.axi4
import framework.generic.bus.chi.snf.LineResponse
import framework.generic.arbiter.RRArbiter
import framework.generic.queue.{Queue => ResponseQueue}
import framework.hart.memory.home.Home
import framework.device.sims.soc.SystemTarget

/**
 * A registered hart supplies its HLink; the simulator owns its external environment. The line memory behind the boot
 * ROM and DDR is the hart's own backing, or, for a hart whose caches are coherent requesters of an external Home, a
 * Home of its own requesters.
 */
abstract class HartTarget(b: GlobalConfig) extends SystemTarget(b) {
  def instantiateHart(): HLinkIO
  def instantiate(b: GlobalConfig): Instance[BaseDevice] =
    Instantiate(new HartEnvironment(b, () => instantiateHart()))
}

@instantiable
class HartEnvironment(b: GlobalConfig, instantiateHart: () => HLinkIO) extends BaseDevice(b.device.ddr.port) {
  val Seq(hartp) = b.harts
  val hart       = instantiateHart()
  // Its line memory is its own backing, a Home of its requesters, or none: a
  // hart that reaches the DDR through its AXI port only.
  require(hart.p.harts == 1 && !(hart.p.backing && hart.p.requesters > 0))
  // A hart alone has no peers: its service endpoint and payload stream idle.
  hart.service.foreach { service =>
    service.request.ready := false.B
    service.reply.valid   := false.B
    service.reply.bits    := 0.U
    service.busy          := false.B
  }
  hart.payload.foreach { payload =>
    payload.send.ready    := false.B
    payload.receive.valid := false.B
    payload.receive.bits  := 0.U
  }
  require(b.device.ddr.clients == 1)

  val peripherals = Instantiate(new Controller(b.device, b.device.ddr.axi, hartp.memory.memoryAccess.memory))

  val decoder = Instantiate(new axi4.Decoder(
    Seq(
      (b.device.clint.base, b.device.clint.bytes),
      (b.device.plic.base, b.device.plic.bytes),
      (b.device.scu.baseAddress, b.device.scu.totalSizeBytes),
      (b.device.trace.base, b.device.trace.bytes)
    ),
    b.device.ddr.axi
  ))

  decoder.io.in <> hart.memory.uncached
  peripherals.io.axi <> decoder.io.matched
  peripherals.io.hardwareInterrupt.sources := 0.U
  hart.interrupts                          := peripherals.io.hardwareInterrupt.harts

  val bridge = Instantiate(new Bridge(b.device.ddr))
  val memory = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(b.device.ddr.port, 3, None)))
  memory.io.in(0) <> hart.memory.ddr
  val widen  = Instantiate(new axi4.Widen(b.device.ddr.axi, b.device.ddr.port, 1))
  widen.io.in <> decoder.io.other
  memory.io.in(1) <> widen.io.out
  memory.io.in(2) <> bridge.io.axi
  dlink.axi <> memory.io.out

  val rom = Instantiate(new BootRomLines(b.device.bootrom, hartp.memory.home.memory.chi))

  if (!hart.p.backing && hart.p.requesters == 0) {
    rom.io.request.valid        := false.B
    rom.io.request.bits         := DontCare
    rom.io.response.ready       := false.B
    bridge.io.request(0).valid  := false.B
    bridge.io.request(0).bits   := DontCare
    bridge.io.response(0).ready := false.B
  } else {
    val (request, response) =
      if (hart.p.backing) (hart.memory.backing.get.request, hart.memory.backing.get.response)
      else {
        // The Home of the hart alone serves its requesters only.
        val homep = hartp.memory.home.copy(memory = hartp.memory.home.memory.copy(agents = hart.p.requesters))
        val home  = Instantiate(new Home(homep, false, hart.memory.page.size))
        home.io.active            := true.B
        home.io.blockRequesterRsp := false.B
        home.io.requesters <> hart.memory.coherent
        home.page.zip(hart.memory.page.toSeq).foreach { case (query, client) => query <> client }
        (home.io.memoryReq, home.io.memoryResp)
      }

    val inRom = request.bits.addr >= b.device.bootrom.base.U &&
      request.bits.addr < (b.device.bootrom.base + b.device.bootrom.bytes).U
    rom.io.request.valid       := request.valid && inRom
    rom.io.request.bits        := request.bits
    bridge.io.request(0).valid := request.valid && !inRom
    bridge.io.request(0).bits  := request.bits
    request.ready              := Mux(inRom, rom.io.request.ready, bridge.io.request(0).ready)
    val answers = Instantiate(new RRArbiter(new LineResponse(hartp.memory.home.memory.chi), 2))
    answers.io.in(0) <> rom.io.response
    answers.io.in(1) <> bridge.io.response(0)
    val queue   = Instantiate(new ResponseQueue(new LineResponse(hartp.memory.home.memory.chi), 1, false, false))
    queue.io.enq <> answers.io.out
    response <> queue.io.deq
  }
}
