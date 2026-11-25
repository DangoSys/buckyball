package examples.threads.matmul

import framework.thread.ant.configs.AntParam

import framework.thread.tbus.ThreadFactory

import framework.thread.configs.ThreadParam

import framework.thread.ant.Ant

import framework.thread.acc.debug.{Debug, DebugIds}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.thread.tlink.{HasTLink, TLinkIO}
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams

@instantiable
class MatMulThread(
  threadp:     ThreadParam,
  permissionp: PmpParams,
  simp:        SimParam)
    extends Module
    with HasTLink {
  // Who the thread is, strapped where it is instantiated: every thread of a hart shares one module.
  @public
  val ids         = IO(Input(new DebugIds))
  val sharedPorts = threadp.acc.get.memDomain.sharedPorts
  val diffTest    = simp.diffTest
  val accp        = threadp.acc.get
  val antp        = AntParam.fromLink(threadp.link)

  val ant: Instance[Ant] =
    Instantiate(new Ant(antp, permissionp))

  val acc: Instance[Acc] = Instantiate(new Acc(accp, permissionp, simp))

  val debug: Instance[Debug] =
    Instantiate(new Debug(accp.frontend, accp.memDomain, accp.gpDomain, accp.ballDomain, simp))

  debug.io.ant   := ant.io.trace
  debug.io.acc <> acc.io.debug
  debug.io.ids   := ids
  acc.io.ownerId := ids.owner

  @public
  val tlink = IO(new TLinkIO(threadp.link, permissionp))

  ant.io.task <> tlink.task
  tlink.tss <> ant.io.shared
  ant.io.rocc <> acc.io.rocc
  tlink.dma <> acc.io.dma

}

object MatMulThread extends ThreadFactory {

  def instantiate(
    threadp:    ThreadParam,
    permission: PmpParams,
    hartId:     Int,
    simp:       SimParam,
    traceId:    Int
  ): TLinkIO = {
    val thread = Instantiate(new MatMulThread(threadp, permission, simp))
    thread.ids.trace       := traceId.U
    thread.ids.owner       := threadp.id.U
    thread.ids.sharedOwner := hartId.U
    thread.tlink
  }

}
