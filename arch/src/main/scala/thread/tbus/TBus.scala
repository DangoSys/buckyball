package framework.thread.tbus

import framework.generic.contract.tlink.Task

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.thread.configs.ThreadParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.sims.configs.SimParam
import framework.generic.dma.Dma
import framework.generic.memory.spm
import framework.thread.tlink.TLinkIO

/** The bus of the thread abstraction: the threads of one hart behind their TLinks. */
@instantiable
class TBus(
  threads:    Seq[ThreadParam],
  permission: PmpParams,
  tss:        spm.Params,
  generators: Seq[() => TLinkIO])
    extends Module {

  @public
  val io = IO(new Bundle {

    val control =
      Option.when(threads.nonEmpty)(MixedVec(threads.map(thread => new Task(thread.link.taskBits, permission))))

    val dma = MixedVec(threads.map(thread => new Dma(thread.link.dma)))

    val tss = Option.when(threads.nonEmpty)(Vec(threads.size, new spm.Port(TBus.this.tss)))
  })

  require(generators.size == threads.size)
  val endpoints = generators.map(_())

  for ((link, index) <- endpoints.zipWithIndex) {
    io.control.get(index) <> link.task
    io.tss.get(index) <> link.tss
    io.dma(index) <> link.dma
  }
}

object TBus {

  def generators(
    threads:    Seq[ThreadParam],
    permission: PmpParams,
    hartId:     Int,
    simp:       SimParam,
    traceIds:   Seq[Int],
    factories:  Map[String, ThreadFactory]
  ): Seq[() => TLinkIO] = {
    require(threads.size == traceIds.size)
    threads.zip(traceIds).map { case (threadp, traceId) =>
      () => factories(threadp.component).instantiate(threadp, permission, hartId, simp, traceId)
    }
  }

}

trait ThreadFactory {

  def instantiate(
    threadp:    ThreadParam,
    permission: PmpParams,
    hartId:     Int,
    simp:       SimParam,
    traceId:    Int
  ): TLinkIO

}
