package framework.thread.tlink

import chisel3._
import framework.thread.configs.ThreadParam
import framework.thread.ant.Task
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.spm

/** Complete thread boundary; each service retains its independent handshake. */
class TLinkIO(thread: ThreadParam, permission: PmpParams) extends Bundle {
  val task = new Task(thread.ant.get, permission)
  val tss  = new spm.Port(thread.ant.get.shared)
  val dma  = new Dma(thread.acc.get.transport.axi)
}
