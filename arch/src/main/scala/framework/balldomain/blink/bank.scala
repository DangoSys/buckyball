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

class BankRead(val b: GlobalConfig) extends Bundle with HasBankId with HasRobId {
  val io = new SramReadIO(b)
}

class BankWrite(val b: GlobalConfig) extends Bundle with HasBankId with HasRobId {
  val io = new SramWriteIO(b)
}

class MultiBankRead(val b: GlobalConfig, BW: Int) extends Bundle with HasBankId with HasRobId {
  val io = Vec(BW, new SramReadIO(b))
}

class MultiBankWrite(val b: GlobalConfig, BW: Int) extends Bundle with HasBankId with HasRobId {
  val io = Vec(BW, new SramWriteIO(b))
}
