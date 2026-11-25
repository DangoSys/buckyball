package framework.generic.acc.memdomain

import framework.generic.acc.memdomain.LayoutConfig
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.memdomain.Response
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class LayoutConfig(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Bundle {
  val vbank_id       = Output(UInt(frontendParam.bank_id_len.W))
  val is_shared      = Output(Bool())
  val is_multi       = Output(Bool())
  val alloc          = Output(Bool())
  val transfer       = Output(Bool())
  val source_bank_id = Output(UInt(frontendParam.bank_id_len.W))
  val clear          = Output(Bool())
  val group_id       = Output(UInt(memParam.groupIdWidth.W))
  val thread_id      = Output(UInt(64.W))
}
