package sims.core

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import _root_.configs.GlobalConfig
import framework.core.clink.CLinkIO
import framework.device.dlink.BaseDevice
import framework.device.ddr.Bridge
import framework.device.peripherals.Controller
import framework.device.peripherals.bootrom.BootRomLines
import framework.ip.bus.axi4
import framework.ip.bus.chi.snf.LineResponse
import framework.lib.arbiter.RRArbiter
import framework.lib.queue.{Queue => ResponseQueue}
import sims.soc.SystemTarget

/** A registered core supplies its CLink; the simulator owns its external environment. */
abstract class CoreTarget(b: GlobalConfig) extends SystemTarget(b) {
  def instantiateCore(): CLinkIO
  def instantiate(b: GlobalConfig): Instance[BaseDevice] =
    Instantiate(new CoreEnvironment(b, () => instantiateCore()))
}

@instantiable
class CoreEnvironment(b: GlobalConfig, instantiateCore: () => CLinkIO) extends BaseDevice(b.device.ddr.axi) {
  val Seq(corep) = b.device.cores
  val core       = instantiateCore()
  require(core.p.harts == 1 && core.p.backing && core.p.requesters == 0)
  require(core.c2c.isEmpty && core.banks.isEmpty)
  require(b.device.ddr.clients == 1)

  val peripherals = Instantiate(new Controller(b.device, b.device.ddr.axi, corep.memory.memoryAccess.memory))

  val decoder = Instantiate(new axi4.Decoder(
    Seq(
      (b.device.clint.base, b.device.clint.bytes),
      (b.device.plic.base, b.device.plic.bytes),
      (b.device.scu.baseAddress, b.device.scu.totalSizeBytes)
    ),
    b.device.ddr.axi
  ))

  decoder.io.in <> core.memory.uncached
  peripherals.io.axi <> decoder.io.matched
  peripherals.io.hardwareInterrupt.sources := 0.U
  core.interrupts                          := peripherals.io.hardwareInterrupt.harts

  val bridge = Instantiate(new Bridge(b.device.ddr))
  val memory = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(b.device.ddr.axi, 3, None)))
  memory.io.in(0) <> core.memory.ddr
  memory.io.in(1) <> decoder.io.other
  memory.io.in(2) <> bridge.io.axi
  dlink.axi <> memory.io.out

  val rom     = Instantiate(new BootRomLines(b.device.bootrom, corep.memory.home.memory.chi))
  val backing = core.memory.backing.get
  val inRom   = backing.request.bits.addr >= b.device.bootrom.base.U &&
    backing.request.bits.addr < (b.device.bootrom.base + b.device.bootrom.bytes).U
  rom.io.request.valid       := backing.request.valid && inRom
  rom.io.request.bits        := backing.request.bits
  bridge.io.request(0).valid := backing.request.valid && !inRom
  bridge.io.request(0).bits  := backing.request.bits
  backing.request.ready      := Mux(inRom, rom.io.request.ready, bridge.io.request(0).ready)
  val answers  = Instantiate(new RRArbiter(new LineResponse(corep.memory.home.memory.chi), 2))
  answers.io.in(0) <> rom.io.response
  answers.io.in(1) <> bridge.io.response(0)
  val response = Instantiate(new ResponseQueue(new LineResponse(corep.memory.home.memory.chi), 1, false, false))
  response.io.enq <> answers.io.out
  backing.response <> response.io.deq
}
