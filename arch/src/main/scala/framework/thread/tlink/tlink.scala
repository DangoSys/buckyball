package framework.thread.tlink

import chisel3._
import framework.thread.configs.ThreadParam
import framework.thread.ant.Task
import framework.core.memory.MemCtrl
import framework.core.memory.access.preflight.configs.PmpParams
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.ip.memory.spm

/** Complete thread boundary; each service retains its independent handshake. */
class TLinkIO(
  thread:     ThreadParam,
  permission: PmpParams,
  tracking:   TrackingParam,
  walker:     WalkerParam)
    extends Bundle {
  val task    = new Task(thread.ant.get, permission)
  val tss     = new spm.Port(thread.ant.get.shared)
  val memCtrl = new MemCtrl(tracking, walker)
  val dma     = new Dma(thread.acc.get.transport.axi)
}
