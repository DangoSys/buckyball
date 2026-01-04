package framework.balldomain.blink

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import chisel3.experimental.hierarchy.{instantiable, public}

class BallStatus extends Bundle {
  val idle    = Output(Bool())
  val running = Output(Bool())
}

// SRAMRead with rob_id, bank_id
class BankRead(val b: GlobalConfig) extends Bundle {
  val io      = new SramReadIO(b)
  val rob_id  = Input(UInt(log2Up(b.frontend.rob_entries).W))
  val bank_id = Input(UInt(log2Up(b.memDomain.bankNum).W))
}

// SRAMWrite with rob_id, bank_id
// wmode is in SramWriteIO.io.req.bits.wmode: true = accumulate, false = overwrite
class BankWrite(val b: GlobalConfig) extends Bundle {
  val io      = new SramWriteIO(b)
  val rob_id  = Input(UInt(log2Up(b.frontend.rob_entries).W))
  val bank_id = Input(UInt(log2Up(b.memDomain.bankNum).W))
}

class BlinkIO(b: GlobalConfig, inBW: Int, outBW: Int) extends Bundle with HasBallStatus {

  def status: BallStatus = new BallStatus()

  val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
  val cmdResp   = Decoupled(new BallRsComplete(b))
  val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
  val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
}

// @instantiable
// class blink(b: GlobalConfig, inBW: Int, outBW: Int) extends Module {
//   @public
//   val io = IO(new BlinkIO(b, inBW, outBW))

//   def cmdReq:    DecoupledIO[BallRsIssue]    = io.cmdReq
//   def cmdResp:   DecoupledIO[BallRsComplete] = io.cmdResp
//   def bankRead:  Vec[BankRead]               = io.bankRead
//   def bankWrite: Vec[BankWrite]              = io.bankWrite
//   def status:    BallStatus                  = io.status
// }
