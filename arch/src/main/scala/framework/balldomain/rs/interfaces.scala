package framework.balldomain.rs

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.decoder.BallDecodeCmd

// Ball domain issue interface - includes global rob_id
class BallRsIssue(b: GlobalConfig) extends Bundle {
  val cmd        = new BallDecodeCmd(b.memDomain.virtualBankCount, b.frontend.iter_len, b.memDomain.groupCountWidth)
  // Global ROB ID
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

// Ball domain completion interface
class BallRsComplete(b: GlobalConfig) extends Bundle {
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}
