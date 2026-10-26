package examples.designs.toy

import sims.configs.SimParam
import framework.core.configs.CoreParam
import framework.device.cbus.CoreFactory
import buckyball.config.{CorePlacement, Design}

import examples.core.rocket.configs.Params
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.clink.{CLinkIO, HasCLink}

/** One RV64GC hart and its local NPU, explicitly wired behind CLink. */
@instantiable
class ToyCore(config: Params, simp: SimParam) extends Module with HasCLink {
  val diffTest = simp.diffTest
  val corep    = config
  require(corep.acc.nonEmpty)
  require(corep.clink.harts == 1 && corep.clink.backing)

  @public
  val clink = IO(new CLinkIO(corep.clink))

  val loom: Instance[Loom] = Instantiate(new Loom(corep))
  val acc:  Instance[Acc]  = Instantiate(new Acc(config, simp))

  loom.io.rocc <> acc.io.rocc
  loom.io.memory <> acc.io.memory
  loom.io.interrupts := clink.interrupts(0)
  clink.memory.uncached <> loom.io.uncached
  clink.memory.backing.get <> loom.io.backing
  clink.memory.ddr <> acc.io.ddr
}

object Toy extends CoreFactory {
  val component = "toy"

  def parameters(design: Design, source: CorePlacement): CoreParam = {
    val native = Params.fromProto(source); CoreParam.fromProto(source, Seq.empty).copy(clink = native.clink)
  }

  def instantiate(
    source:   CorePlacement,
    corep:    CoreParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): CLinkIO =
    Instantiate(new ToyCore(Params.fromProto(source), simp)).clink

}
