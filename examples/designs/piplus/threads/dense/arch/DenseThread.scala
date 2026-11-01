package examples.designs.piplus.threads.dense

import examples.designs.piplus.thread.Acc
import framework.core.tbus.ThreadFactory
import framework.core.configs.CoreParam
import sims.configs.SimParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.thread.configs.ThreadParam
import framework.thread.ant.Ant
import framework.thread.tlink.{HasTLink, Stream, TLinkIO}
import framework.acc.debug.Debug

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

/** The dense thread: an Ant hart issuing the MXINT8 Ball instructions to its accelerator, whose DMA goes to DDR. */
@instantiable
class DenseThread(
  threadp:       ThreadParam,
  permissionp:   PmpParams,
  trackingp:     TrackingParam,
  walkerp:       WalkerParam,
  simp:          SimParam,
  traceId:       Int,
  sharedOwnerId: Int)
    extends Module
    with HasTLink {
  val accp = threadp.acc.get

  @public
  val tlink = IO(new TLinkIO(threadp, permissionp, trackingp, walkerp))

  val ant: Instance[Ant] = Instantiate(new Ant(threadp.ant.get, permissionp))
  val acc: Instance[Acc] = Instantiate(new Acc(accp, permissionp, trackingp, walkerp, simp))

  val debug: Instance[Debug] =
    Instantiate(new Debug(accp.frontend, accp.memDomain, accp.gpDomain, accp.ballDomain, simp, traceId, sharedOwnerId))
  debug.io.ant := ant.io.trace
  debug.io.acc <> acc.io.debug

  ant.io.task <> tlink.task
  tlink.tss <> ant.io.shared
  ant.io.rocc <> acc.io.rocc
  tlink.memCtrl <> acc.io.memory

  val transport = Instantiate(new Stream(accp.transport.axi))
  transport.io.acc <> acc.io.ddr
  transport.io.readAddress  := acc.io.readAddress
  transport.io.writeAddress := acc.io.writeAddress
  tlink.dma <> transport.io.dma
}

object DenseThread extends ThreadFactory {

  def instantiate(
    corep:   CoreParam,
    threadp: ThreadParam,
    simp:    SimParam,
    traceId: Int
  ): TLinkIO =
    Instantiate(new DenseThread(
      threadp,
      corep.privileged.permission,
      corep.memory.dcache.tracking,
      corep.memory.memoryAccess.walker,
      simp,
      traceId,
      corep.hartId
    )).tlink

}
