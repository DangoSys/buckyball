package framework.generic.contract.memdomain

import framework.generic.contract.blink.BlinkParam

import framework.generic.contract.memory.access.mmu.configs.{Params => WalkerParam}
import framework.generic.contract.shm.LocalBank

import framework.generic.contract.memdomain.MvoverISA
import framework.generic.contract.memdomain.MvoverPort

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
import framework.generic.contract.frontend.scheduler.MemoryIssue
import framework.generic.contract.ball.DomainMemory
import framework.generic.contract.memdomain.BankLayout
import framework.generic.contract.memdomain.BankLayoutRequest
import framework.generic.contract.memdomain.BankPort
import framework.generic.contract.memdomain.BankRead
import framework.generic.contract.memdomain.BankWrite
import framework.generic.contract.memdomain.Response
import framework.generic.contract.memdomain.DmaFaultControl

class BankRead(link: BlinkParam) extends Bundle {
  val bank_id  = Input(UInt(link.bankIdBits.W))
  val rob_id   = Input(UInt(link.robIdBits.W))
  val group_id = Input(UInt(link.groupIdBits.W))

  val io = new SramReadIO(link.bankWidth)
}

class BankWrite(link: BlinkParam) extends Bundle {
  val bank_id  = Input(UInt(link.bankIdBits.W))
  val rob_id   = Input(UInt(link.robIdBits.W))
  val group_id = Input(UInt(link.groupIdBits.W))

  val io = new SramWriteIO(link.bankWidth, link.bankMaskLen)
}

class BankPort(link: BlinkParam) extends Bundle {
  val write     = Flipped(new SramWriteIO(link.bankWidth, link.bankMaskLen))
  val read      = Flipped(new SramReadIO(link.bankWidth))
  val bank_id   = Output(UInt(link.bankIdBits.W))
  val group_id  = Output(UInt(link.groupIdBits.W))
  val is_shared = Output(Bool())
  val thread_id = Output(UInt(64.W))
  val rob_id    = Output(UInt(link.robIdBits.W))
  val inst_id   = Output(UInt(64.W))
}

class BankLayoutRequest(bankIdBits: Int) extends Bundle {
  val banks   = Vec(3, UInt(bankIdBits.W))
  val enabled = Vec(3, Bool())
}

class BankLayout(groupCountBits: Int) extends Bundle {
  val groups = Vec(3, UInt(groupCountBits.W))
}
