package framework.thread.tlink

import framework.generic.contract.tlink.TLinkParam

import framework.generic.contract.tlink.Task

import chisel3._
import framework.generic.memory.spm
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.dma.Dma

/** Complete thread boundary; each service retains its independent handshake. */
class TLinkIO(p: TLinkParam, permission: PmpParams) extends Bundle {
  val task = new Task(p.taskBits, permission)
  val tss  = new spm.Port(p.tss)
  val dma  = new Dma(p.dma)
}
