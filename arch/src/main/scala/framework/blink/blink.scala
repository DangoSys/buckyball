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

// Ball设备的标准接口
class Blink(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmdReq = Flipped(Decoupled(new BallRsIssue))
  val cmdResp = Decoupled(new BallRsComplete)

  val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
  val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

  val status = new Status
}
