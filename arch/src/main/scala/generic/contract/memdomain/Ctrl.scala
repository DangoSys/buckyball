package framework.generic.contract.memdomain

import framework.generic.arbiter.{Arbiter}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.scheduler.MemoryIssue
import framework.generic.contract.memdomain.toddr.DmaStatus
import framework.generic.contract.memdomain.Response

class Response(robIdBits: Int) extends Bundle {
  val fault  = new DmaStatus
  val rob_id = UInt(robIdBits.W)
}
