package framework.blink

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.memdomain.mem.{SramReadIO, SramWriteIO}

// Ball device status bundle
class Status extends Bundle {
  // device is ready to accept new input
  val ready = Output(Bool())
  // device has valid output
  val valid = Output(Bool())
  // no input and no output
  val idle = Output(Bool())
  // has input but no output
  val init = Output(Bool())
  // started producing output
  val running = Output(Bool())
  // fully finished current batch
  val complete = Output(Bool())
  // current batch iteration
  val iter = Output(UInt(32.W))
}

// SramReadIO with rob_id
class SramReadWithRobId(val n: Int, val w: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val io = new SramReadIO(n, w)
  // Input because the outer layer has Flipped
  val rob_id = Input(UInt(log2Up(b.rob_entries).W))
}

// SramWriteIO with rob_id
class SramWriteWithRobId(val n: Int, val w: Int, val mask_len: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val io = new SramWriteIO(n, w, mask_len)
  // Input because the outer layer has Flipped
  val rob_id = Input(UInt(log2Up(b.rob_entries).W))
}

// Standard interface for Ball devices
class Blink(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)

  val sramRead = Vec(b.sp_banks, Flipped(new SramReadWithRobId(b.spad_bank_entries, b.spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteWithRobId(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  val accRead = Vec(b.acc_banks, Flipped(new SramReadWithRobId(b.acc_bank_entries, b.acc_w)))
  val accWrite = Vec(b.acc_banks, Flipped(new SramWriteWithRobId(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

  val status = new Status
}
