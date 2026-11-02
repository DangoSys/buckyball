package examples.designs.piplus.threads.action

import examples.designs.piplus.thread.Acc
import framework.core.tbus.ThreadFactory
import framework.core.configs.CoreParam
import sims.configs.SimParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.thread.configs.ThreadParam
import framework.thread.ant.Ant
import framework.thread.tlink.{HasTLink, Stream, TLinkIO}
import framework.acc.debug.Debug

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

/** The action thread: an Ant hart issuing the MXINT8 Ball instructions to its accelerator, whose DMA goes to DDR. */
@instantiable
class ActionThread(
  threadp:       ThreadParam,
  permissionp:   PmpParams,
  simp:          SimParam,
  traceId:       Int,
  sharedOwnerId: Int)
    extends Module
    with HasTLink {
  val accp = threadp.acc.get

  @public
  val tlink = IO(new TLinkIO(threadp, permissionp))

  val ant: Instance[Ant] = Instantiate(new Ant(threadp.ant.get, permissionp))
  val acc: Instance[Acc] = Instantiate(new Acc(accp, permissionp, simp))

  val debug: Instance[Debug] =
    Instantiate(new Debug(accp.frontend, accp.memDomain, accp.gpDomain, accp.ballDomain, simp, traceId, sharedOwnerId))
  debug.io.ant := ant.io.trace
  debug.io.acc <> acc.io.debug

  ant.io.task <> tlink.task
  tlink.tss <> ant.io.shared
  ant.io.rocc <> acc.io.rocc
  tlink.dma <> acc.io.dma
}

object ActionThread extends ThreadFactory {

  def instantiate(
    corep:   CoreParam,
    threadp: ThreadParam,
    simp:    SimParam,
    traceId: Int
  ): TLinkIO =
    Instantiate(new ActionThread(
      threadp,
      corep.privileged.permission,
      simp,
      traceId,
      corep.hartId
    )).tlink

}
