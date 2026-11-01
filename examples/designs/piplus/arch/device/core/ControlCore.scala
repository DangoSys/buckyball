package examples.designs.piplus

import sims.configs.SimParam
import framework.device.cbus.CoreFactory
import buckyball.config.{CorePlacement, Design}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.controller.Controller
import framework.core.clink.{CLinkIO, HasCLink}

/** The control core: one hart for the runtime, tokenization, Euler update and scheduling. */
@instantiable
class ControlCore(corep: CoreParam) extends Module with HasCLink {
  require(corep.clink.harts == 1 && !corep.clink.backing && !corep.clink.service)
  require(corep.clink.requesters == 2 && corep.threads.isEmpty && corep.acc.isEmpty)

  @public
  val clink = IO(new CLinkIO(corep.clink))

  val loom:       Instance[Loom]       = Instantiate(new Loom(corep))
  val controller: Instance[Controller] = Instantiate(new Controller(corep.controller))

  loom.io.interrupts                  := clink.interrupts(0)
  controller.io.execution <> loom.io.commands
  loom.io.drained                     := controller.io.drained
  loom.io.memoryAllowed               := controller.io.memoryAllowed
  controller.io.move.command.ready    := false.B
  controller.io.move.completion.valid := false.B
  controller.io.move.completion.bits  := false.B

  clink.memory.coherent.zip(loom.io.coherent).foreach { case (port, cache) => port <> cache }
  clink.memory.uncached <> loom.io.uncached
  clink.memory.ddr.aw.valid := false.B
  clink.memory.ddr.aw.bits  := 0.U.asTypeOf(clink.memory.ddr.aw.bits)
  clink.memory.ddr.w.valid  := false.B
  clink.memory.ddr.w.bits   := 0.U.asTypeOf(clink.memory.ddr.w.bits)
  clink.memory.ddr.ar.valid := false.B
  clink.memory.ddr.ar.bits  := 0.U.asTypeOf(clink.memory.ddr.ar.bits)
  clink.memory.ddr.b.ready  := false.B
  clink.memory.ddr.r.ready  := false.B
}

object Control extends CoreFactory {
  val component = "control"

  def parameters(design: Design, source: CorePlacement): CoreParam =
    CoreParam.fromProto(design, source, externalHome = true)

  def instantiate(
    source:   CorePlacement,
    corep:    CoreParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): CLinkIO =
    Instantiate(new ControlCore(corep)).clink

}
