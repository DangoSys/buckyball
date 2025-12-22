package framework.balldomain.blink

import chisel3._
import chisel3.util._
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}

// Ball device status bundle
class Status extends Bundle {
  val ready    = Output(Bool())
  val valid    = Output(Bool())
  val idle     = Output(Bool())
  val init     = Output(Bool())
  val running  = Output(Bool())
  val complete = Output(Bool())
  val iter     = Output(UInt(32.W))
}

// BankRead with rob_id, bank_id
class BankRead(
  val n:       Int,
  val w:       Int,
  rob_entries: Int,
  total_banks: Int)
    extends Bundle {
  val io      = new SramReadIO(n, w)
  // Input because the outer layer has Flipped
  val rob_id  = Input(UInt(log2Up(rob_entries).W))
  val bank_id = Input(UInt(log2Up(total_banks).W))
}

// BankWrite with rob_id, bank_id
// wmode is in SramWriteIO.io.req.bits.wmode: true = accumulate (累加), false = overwrite (覆盖)
class BankWrite(
  val n:        Int,
  val w:        Int,
  val mask_len: Int,
  rob_entries:  Int,
  total_banks:  Int)
    extends Bundle {
  val io      = new SramWriteIO(n, w, mask_len)
  // Input because the outer layer has Flipped
  val rob_id  = Input(UInt(log2Up(rob_entries).W))
  val bank_id = Input(UInt(log2Up(total_banks).W))
}

// Standard interface for Ball devices
// bankEntries, bankWidth, bankMaskLen come from MemDomain, not BallDomain
class Blink(
  parameter:   BallDomainParam,
  bankEntries: Int,
  bankWidth:   Int,
  bankMaskLen: Int)
    extends Bundle {
  val cmdReq  = Flipped(Decoupled(new BallRsIssue(parameter)))
  val cmdResp = Decoupled(new BallRsComplete(parameter))

  val bankRead =
    Vec(parameter.numBanks, Flipped(new BankRead(bankEntries, bankWidth, parameter.rob_entries, parameter.numBanks)))

  val bankWrite = Vec(
    parameter.numBanks,
    Flipped(new BankWrite(bankEntries, bankWidth, bankMaskLen, parameter.rob_entries, parameter.numBanks))
  )

  val status = new Status
}
