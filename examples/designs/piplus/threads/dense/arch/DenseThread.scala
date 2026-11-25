package examples.designs.piplus.threads.dense

import framework.thread.ant.configs.AntParam

import framework.thread.tbus.ThreadFactory

import framework.thread.configs.ThreadParam

import framework.thread.ant.Ant
import framework.thread.tlink.{HasTLink, TLinkIO}
import framework.thread.acc.debug.{Debug, DebugIds}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams

/** The dense thread: an Ant hart issuing the MXINT8 Ball instructions to its accelerator, whose DMA goes to DDR. */
@instantiable
class DenseThread(
  threadp:     ThreadParam,
  permissionp: PmpParams,
  simp:        SimParam)
    extends Module
    with HasTLink {
  // Who the thread is, strapped where it is instantiated: every thread of a hart shares one module.
  @public
  val ids  = IO(Input(new DebugIds))
  val accp = threadp.acc.get

  @public
  val tlink = IO(new TLinkIO(threadp.link, permissionp))

  val ant: Instance[Ant] = Instantiate(new Ant(AntParam.fromLink(threadp.link), permissionp))
  val acc: Instance[Acc] = Instantiate(new Acc(accp, permissionp, simp))

  val debug: Instance[Debug] =
    Instantiate(new Debug(accp.frontend, accp.memDomain, accp.gpDomain, accp.ballDomain, simp))
  debug.io.ant   := ant.io.trace
  debug.io.acc <> acc.io.debug
  debug.io.ids   := ids
  acc.io.ownerId := ids.owner

  ant.io.task <> tlink.task
  tlink.tss <> ant.io.shared
  ant.io.rocc <> acc.io.rocc
  tlink.dma <> acc.io.dma
}

object DenseThread extends ThreadFactory {

  def instantiate(
    threadp:    ThreadParam,
    permission: PmpParams,
    hartId:     Int,
    simp:       SimParam,
    traceId:    Int
  ): TLinkIO = {
    val thread = Instantiate(new DenseThread(threadp, permission, simp))
    thread.ids.trace       := traceId.U
    thread.ids.owner       := threadp.id.U
    thread.ids.sharedOwner := hartId.U
    thread.tlink
  }

}
