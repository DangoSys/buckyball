package framework.thread.acc.memdomain

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.memory.access.mmu.configs.{Params => WalkerParam}
import framework.generic.contract.shm.LocalBank
import framework.thread.acc.shm.BankAccess
import framework.thread.acc.memdomain.LayoutQuery
import framework.thread.acc.memdomain.tomem.configs.{Params => TransportParam}
import framework.generic.contract.memdomain.MvoverISA
import framework.generic.contract.memdomain.MvoverPort
import framework.thread.acc.memdomain.PhysicalBankHash
import framework.generic.arbiter.{RRArbiter}
import framework.generic.memory.interlock.{Params => TrackingParam}
import framework.generic.memory.bank.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq, SramWriteResp}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.bus.axi4
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.configs.BallDomainParam
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.gpdomain.configs.GpDomainParam
import framework.thread.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.frontend.scheduler.MemoryIssue
import framework.generic.contract.ball.DomainMemory
import framework.generic.contract.memdomain.BankLayout
import framework.generic.contract.memdomain.BankLayoutRequest
import framework.generic.contract.memdomain.BankPort
import framework.generic.contract.memdomain.BankRead
import framework.generic.contract.memdomain.BankWrite
import framework.generic.contract.memdomain.Response
import framework.generic.contract.memdomain.DmaFaultControl
import framework.generic.contract.memdomain._

class PhysicalBankHash(val frontendParam: FrontendParam) extends Bundle {
  val valid      = Bool()
  val threadId   = UInt(64.W)
  val vbankId    = UInt(frontendParam.bank_id_len.W)
  val pbankId    = UInt(32.W)
  val groupId    = UInt(32.W)
  val statusHash = UInt(32.W)
}

class LayoutQuery(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val bank   = Output(UInt(frontendParam.bank_id_len.W))
  val groups = Input(UInt(memParam.groupCountWidth.W))
}
