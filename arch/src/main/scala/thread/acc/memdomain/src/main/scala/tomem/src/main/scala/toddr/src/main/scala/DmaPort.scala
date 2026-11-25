package framework.thread.acc.memdomain.toddr

import framework.generic.contract.memory.access.preflight.Params

import framework.thread.acc.memdomain.tomem.configs.{Params => ModuleParam}
import framework.generic.contract.memdomain.MemoryContext

import chisel3._
import chisel3.util._
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.memdomain.toddr.DmaStatus

class DmaDecision(b: ModuleParam) extends Bundle {
  val p = b.mapping

  val valid    = Bool()
  val parentId = UInt(p.idBits.W)
  val fault    = new DmaStatus
}
