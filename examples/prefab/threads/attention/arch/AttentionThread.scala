package examples.threads.attention

import framework.thread.ant.Ant
import framework.core.tbus.ThreadFactory
import framework.core.configs.CoreParam
import sims.configs.SimParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.thread.configs.ThreadParam
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}

import framework.acc.debug.Debug

import chisel3.experimental.hierarchy.Instance

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.thread.tlink.{HasTLink, TLinkIO}

@instantiable
class AttentionThread(
  threadp:       ThreadParam,
  permissionp:   PmpParams,
  trackingp:     TrackingParam,
  walkerp:       WalkerParam,
  simp:          SimParam,
  traceId:       Int,
  sharedOwnerId: Int)
    extends Module
    with HasTLink {
  val sharedPorts = threadp.acc.get.memDomain.sharedPorts
  val diffTest    = simp.diffTest
  val accp        = threadp.acc.get
  val antp        = threadp.ant.get

  val ant: Instance[Ant] =
    Instantiate(new Ant(antp, permissionp))

  val acc: Instance[Acc] = Instantiate(new Acc(accp, permissionp, trackingp, walkerp, simp))

  val debug: Instance[Debug] =
    Instantiate(new Debug(accp.frontend, accp.memDomain, accp.gpDomain, accp.ballDomain, simp, traceId, sharedOwnerId))

  debug.io.ant := ant.io.trace
  debug.io.acc <> acc.io.debug

  @public
  val tlink = IO(new TLinkIO(threadp, permissionp, trackingp, walkerp, simp))

  ant.io.task <> tlink.task
  tlink.tss <> ant.io.shared
  ant.io.rocc <> acc.io.rocc
  tlink.memory <> acc.io.memory
  tlink.ddr <> acc.io.ddr
  tlink.shm <> acc.io.shm
  acc.io.shared_bank_hashes.zip(tlink.sharedHashes).foreach { case (port, hashes) => port := hashes }
}

object AttentionThread extends ThreadFactory {

  def instantiate(
    corep:   CoreParam,
    threadp: ThreadParam,
    simp:    SimParam,
    traceId: Int
  ): TLinkIO =
    Instantiate(new AttentionThread(
      threadp,
      corep.privileged.permission,
      corep.memory.dcache.tracking,
      corep.memory.memoryAccess.walker,
      simp,
      traceId,
      corep.hartId
    )).tlink

}
