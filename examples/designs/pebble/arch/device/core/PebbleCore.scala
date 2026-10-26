package examples.designs.pebble

import sims.configs.SimParam
import framework.device.cbus.CoreFactory
import buckyball.config.{CorePlacement, Design}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.clink.{CLinkIO, HasCLink}

/** One RV64GC hart and its local NPU, explicitly wired behind CLink. */
@instantiable
class PebbleCore(corep: CoreParam, simp: SimParam) extends Module with HasCLink {
  val diffTest = simp.diffTest
  require(corep.threads.isEmpty && corep.acc.nonEmpty, "PebbleCore connects a local ACC without TBus")
  require(corep.clink.harts == 1 && corep.clink.backing)

  @public
  val clink = IO(new CLinkIO(corep.clink))

  val loom: Instance[Loom] = Instantiate(new Loom(corep))
  val acc:  Instance[Acc]  = Instantiate(new Acc(corep, corep.acc.get, simp))

  loom.io.rocc <> acc.io.rocc
  loom.io.memory <> acc.io.memory
  loom.io.interrupts := clink.interrupts(0)
  clink.memory.uncached <> loom.io.uncached
  clink.memory.backing.get <> loom.io.backing
  clink.memory.ddr <> acc.io.ddr
}

object Pebble extends CoreFactory {
  val component = "pebble"
  def parameters(design: Design, source: CorePlacement): CoreParam =
    CoreParam.fromProto(design, source, externalHome = false)

  def instantiate(
    source:   CorePlacement,
    corep:    CoreParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): CLinkIO =
    Instantiate(new PebbleCore(corep, simp)).clink

}
