package framework.balldomain.blink

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import chisel3.experimental.hierarchy.{instantiable, public}

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
class BankRead(val b: GlobalConfig) extends Bundle {
  val io      = new SramReadIO(b)
  val rob_id  = Input(UInt(log2Up(b.frontend.rob_entries).W))
  val bank_id = Input(UInt(log2Up(b.memDomain.bankNum).W))
}

// BankWrite with rob_id, bank_id
// wmode is in SramWriteIO.io.req.bits.wmode: true = accumulate, false = overwrite
class BankWrite(val b: GlobalConfig) extends Bundle {
  val io      = new SramWriteIO(b)
  val rob_id  = Input(UInt(log2Up(b.frontend.rob_entries).W))
  val bank_id = Input(UInt(log2Up(b.memDomain.bankNum).W))
}

// Blink IO Bundle - interface for Ball devices
// inBW: number of input bandwidth channels (bankRead)
// outBW: number of output bandwidth channels (bankWrite)
class BlinkIO(b: GlobalConfig, inBW: Int, outBW: Int) extends Bundle {
  val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
  val cmdResp   = Decoupled(new BallRsComplete(b))
  val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
  val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
  val status    = new Status
}

@instantiable
class Blink(b: GlobalConfig, inBW: Int, outBW: Int) extends Module {
  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def cmdReq:    DecoupledIO[BallRsIssue]    = io.cmdReq
  def cmdResp:   DecoupledIO[BallRsComplete] = io.cmdResp
  def bankRead:  Vec[BankRead]               = io.bankRead
  def bankWrite: Vec[BankWrite]              = io.bankWrite
  def status:    Status                      = io.status
}
