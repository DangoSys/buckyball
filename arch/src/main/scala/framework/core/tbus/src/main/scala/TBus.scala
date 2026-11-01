package framework.core.tbus

import sims.configs.SimParam
import framework.core.tbus.configs.Layout

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.thread.tlink.{Dma, TLinkIO}
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.thread.configs.ThreadParam
import framework.acc.memdomain.{MvoverPort, PhysicalBankHash}
import framework.core.controller.admission.TaskEndpoint
import framework.thread.ant.Task
import framework.core.memory.MemCtrl
import framework.ip.memory.spm
import framework.core.tbus.SharedNetwork
import framework.ip.bus.axi4
import framework.ip.bus.axi4.Interconnect

@instantiable
class TBus(
  corep:      CoreParam,
  simp:       SimParam,
  traceIds:   Seq[Int],
  generators: Seq[() => TLinkIO])
    extends Module {
  val diffTest = simp.diffTest
  val core     = corep
  require(traceIds.size == core.threads.size)

  @public
  val io = IO(new Bundle {

    val control = Option.when(core.threads.nonEmpty)(MixedVec(core.threads.map(thread =>
      new Task(thread.ant.get, core.privileged.permission)
    )))

    val memory = Option.when(core.threads.nonEmpty)(Vec(
      core.threads.size,
      new MemCtrl(core.memory.dcache.tracking, core.memory.memoryAccess.walker)
    ))

    val dma = MixedVec(core.threads.map(thread => new Dma(thread.acc.get.transport.axi)))

    val tss = Option.when(core.threads.nonEmpty)(Vec(core.threads.size, new spm.Port(core.tss.get.memory)))
  })

  require(generators.size == core.threads.size)
  val endpoints = generators.map(_())

  require(core.tss.get.ports == endpoints.size)
  require(core.interconnect.masters == endpoints.size)
  require(core.tbus.get.participants == core.threads.map(_.id) && core.tbus.get.slots == endpoints.size)

  for ((link, index) <- endpoints.zipWithIndex) {
    io.control.get(index) <> link.task
    io.tss.get(index) <> link.tss
    io.memory.get(index) <> link.memCtrl
    io.dma(index) <> link.dma
  }
}

object TBus {

  def generators(
    corep:     CoreParam,
    simp:      SimParam,
    traceIds:  Seq[Int],
    factories: Map[String, ThreadFactory]
  ): Seq[() => TLinkIO] = {
    require(corep.threads.size == traceIds.size)
    corep.threads.zip(traceIds).map { case (threadp, traceId) =>
      () => factories(threadp.component).instantiate(corep, threadp, simp, traceId)
    }
  }

}

trait ThreadFactory {

  def instantiate(
    corep:   CoreParam,
    threadp: ThreadParam,
    simp:    SimParam,
    traceId: Int
  ): TLinkIO

}
