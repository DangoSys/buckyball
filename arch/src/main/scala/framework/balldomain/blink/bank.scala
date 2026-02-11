package framework.balldomain.blink

import chisel3._
import chisel3.util._
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.memdomain.backend.banks.{SramReadReq, SramWriteReq}
import framework.top.GlobalConfig

trait HasBankId {
  val b: GlobalConfig
  val bank_id = Input(UInt(log2Up(b.memDomain.bankNum).W))
}

trait HasRobId {
  val b: GlobalConfig
  val rob_id = Input(UInt(log2Up(b.frontend.rob_entries).W))
}

trait HasBallId {
  val b: GlobalConfig
  val ball_id = Input(UInt(log2Up(b.ballDomain.ballNum).W))
}

trait HasAccGroupId {
  val b: GlobalConfig
  val acc_group_id = Input(UInt(3.W))
}

class BankRead(val b: GlobalConfig) extends Bundle with HasBankId with HasRobId with HasBallId with HasAccGroupId {
  val io = new SramReadIO(b)
}

class BankWrite(val b: GlobalConfig) extends Bundle with HasBankId with HasRobId with HasBallId with HasAccGroupId {
  val io = new SramWriteIO(b)
}
