package examples.designs.pebble.harts.pebble

import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.generic.contract.sims.configs.SimParam

/** One RV64GC hart and its local NPU, explicitly wired behind HLink. */
@instantiable
class PebbleHart(hartp: HartParam, simp: SimParam) extends Module with HasHLink {
  val diffTest = simp.diffTest
  require(hartp.threads.isEmpty && hartp.acc.nonEmpty, "PebbleHart connects a local ACC without TBus")
  require(hartp.hlink.harts == 1 && hartp.hlink.backing)

  @public
  val hlink = IO(new HLinkIO(hartp.hlink))

  val loom: Instance[Loom] = Instantiate(new Loom(hartp))
  val acc:  Instance[Acc]  = Instantiate(new Acc(hartp, hartp.acc.get, simp))

  loom.io.rocc <> acc.io.rocc
  loom.io.memory <> acc.io.memory
  loom.io.fault <> acc.io.fault
  loom.io.interrupts := hlink.interrupts(0)
  hlink.memory.uncached <> loom.io.uncached
  hlink.memory.backing.get <> loom.io.backing
  hlink.memory.ddr <> acc.io.ddr
}

object Pebble extends HartFactory {
  val component = "pebble"
  def parameters(design: Design, source: HartPlacement): HartParam =
    HartParam.fromProto(design, source, externalHome = false)

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new PebbleHart(hartp, simp)).hlink

}
