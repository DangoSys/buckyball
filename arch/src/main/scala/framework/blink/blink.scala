package framework.blink

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}

// Ball device status bundle
class Status extends Bundle {
  val ready = Output(Bool())      // device is ready to accept new input
  val valid = Output(Bool())      // device has valid output
  val idle = Output(Bool())       // no input and no output
  val init = Output(Bool())       // has input but no output
  val running = Output(Bool())    // started producing output
  val complete = Output(Bool())   // fully finished current batch
  val iter = Output(UInt(32.W))  // current batch iteration
}

// SramReadIO 包含 rob_id
class SramReadWithRobId(val n: Int, val w: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val io = new SramReadIO(n, w)
  val rob_id = Input(UInt(log2Up(b.rob_entries).W))  // Input 因为外层有 Flipped
}

// SramWriteIO 包含 rob_id
class SramWriteWithRobId(val n: Int, val w: Int, val mask_len: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val io = new SramWriteIO(n, w, mask_len)
  val rob_id = Input(UInt(log2Up(b.rob_entries).W))  // Input 因为外层有 Flipped
}

// Ball设备的标准接口
class Blink(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)

  val sramRead = Vec(b.sp_banks, Flipped(new SramReadWithRobId(b.spad_bank_entries, b.spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteWithRobId(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  val accRead = Vec(b.acc_banks, Flipped(new SramReadWithRobId(b.acc_bank_entries, b.acc_w)))
  val accWrite = Vec(b.acc_banks, Flipped(new SramWriteWithRobId(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

  val status = new Status
}
