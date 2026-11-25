package examples.designs.toy.harts.toy

import framework.hart.configs.HartParam
import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import examples.hart.rocket.configs.Params
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.generic.contract.sims.configs.SimParam

/** One RV64GC hart and its local NPU, explicitly wired behind HLink. */
@instantiable
class ToyHart(config: Params, simp: SimParam) extends Module with HasHLink {
  val diffTest = simp.diffTest
  val hartp    = config
  require(hartp.acc.nonEmpty)
  require(hartp.hlink.harts == 1 && hartp.hlink.backing)

  @public
  val hlink = IO(new HLinkIO(hartp.hlink))

  val loom: Instance[Loom] = Instantiate(new Loom(hartp))
  val acc:  Instance[Acc]  = Instantiate(new Acc(config, simp))

  loom.io.rocc <> acc.io.rocc
  loom.io.memory <> acc.io.memory
  loom.io.fault <> acc.io.fault
  loom.io.interrupts := hlink.interrupts(0)
  hlink.memory.uncached <> loom.io.uncached
  hlink.memory.backing.get <> loom.io.backing
  hlink.memory.ddr <> acc.io.ddr
}

object Toy extends HartFactory {
  val component = "toy"

  def parameters(design: Design, source: HartPlacement): HartParam = {
    val native = Params.fromProto(source); HartParam.fromProto(source, Seq.empty).copy(hlink = native.hlink)
  }

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new ToyHart(Params.fromProto(source), simp)).hlink

}
