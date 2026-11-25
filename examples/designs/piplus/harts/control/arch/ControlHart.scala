package examples.designs.piplus.harts.control

import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.hart.memory.{Requester}
import framework.generic.contract.sims.configs.SimParam

/**
 * The control hart: one hart for the runtime, tokenization, Euler update and scheduling. It has no accelerator and no
 * threads; the DDR port of its link is unused.
 */
@instantiable
class ControlHart(hartp: HartParam) extends Module with HasHLink {
  require(hartp.hlink.harts == 1 && !hartp.hlink.backing && !hartp.hlink.service)
  require(hartp.hlink.requesters == 1 && hartp.threads.isEmpty && hartp.acc.isEmpty)

  @public
  val hlink = IO(new HLinkIO(hartp.hlink))

  val loom: Instance[Loom] = Instantiate(new Loom(hartp))

  loom.io.interrupts := hlink.interrupts(0)
  hlink.memory.uncached <> loom.io.uncached

  val requester = Instantiate(new Requester(
    hartp.hlink.line,
    hartp.hlink.requesterIds.head,
    Seq(hartp.memory.icache.nodeId, hartp.memory.dcache.cache.nodeId),
    2
  ))

  requester.io.clients.take(2).zip(loom.io.coherent).foreach { case (port, cache) => port <> cache }
  hlink.memory.coherent(0) <> requester.io.coherent

  hlink.memory.ddr.aw.valid := false.B
  hlink.memory.ddr.aw.bits  := 0.U.asTypeOf(hlink.memory.ddr.aw.bits)
  hlink.memory.ddr.w.valid  := false.B
  hlink.memory.ddr.w.bits   := 0.U.asTypeOf(hlink.memory.ddr.w.bits)
  hlink.memory.ddr.ar.valid := false.B
  hlink.memory.ddr.ar.bits  := 0.U.asTypeOf(hlink.memory.ddr.ar.bits)
  hlink.memory.ddr.b.ready  := true.B
  hlink.memory.ddr.r.ready  := true.B
}

object Control extends HartFactory {
  val component = "control"

  def parameters(design: Design, source: HartPlacement): HartParam =
    HartParam.fromProto(design, source, externalHome = true)

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new ControlHart(hartp)).hlink

}
