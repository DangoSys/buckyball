package examples.designs.piplus.harts.solo

import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.hart.memory.{Requester}
import framework.generic.contract.sims.configs.SimParam

/**
 * The solo hart: one hart with the dense thread's accelerator on its RoCC port and no thread, for work and tuning on a
 * single hart; the accelerator's DMA is a coherent requester of the hart's node. It is not part of the device.
 */
@instantiable
class SoloHart(hartp: HartParam, simp: SimParam) extends Module with HasHLink {
  val accp = hartp.acc.get
  require(hartp.hlink.harts == 1 && !hartp.hlink.backing && !hartp.hlink.service)
  require(hartp.hlink.requesters == 1 && hartp.threads.isEmpty)

  @public
  val hlink = IO(new HLinkIO(hartp.hlink))

  val loom: Instance[Loom] = Instantiate(new Loom(hartp))
  val acc:  Instance[Acc]  = Instantiate(new Acc(hartp, accp, simp))

  loom.io.interrupts := hlink.interrupts(0)
  loom.io.rocc <> acc.io.rocc
  loom.io.memory <> acc.io.memory
  loom.io.fault <> acc.io.fault
  hlink.memory.uncached <> loom.io.uncached

  val requester = Instantiate(new Requester(
    hartp.hlink.line,
    hartp.hlink.requesterIds.head,
    Seq(hartp.memory.icache.nodeId, hartp.memory.dcache.cache.nodeId),
    2
  ))

  requester.io.clients.take(2).zip(loom.io.coherent).foreach { case (port, cache) => port <> cache }
  hlink.memory.coherent(0) <> requester.io.coherent

  val widen = Instantiate(new framework.generic.bus.axi4.Widen(accp.transport.axi, hartp.hlink.ddr, 256))
  widen.io.in <> acc.io.ddr
  hlink.memory.ddr <> widen.io.out
}

object Solo extends HartFactory {
  val component = "solo"

  def parameters(design: Design, source: HartPlacement): HartParam =
    HartParam.fromProto(design, source, externalHome = true)

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new SoloHart(hartp, simp)).hlink

}
