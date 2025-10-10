package prototype.nagisa.gelu

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

// Control -> Load 接口
class CtrlLdReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val op1_bank      = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val op1_bank_addr = UInt(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W)
  val iter          = UInt(10.W)
  val is_acc        = Bool()
}

// Control -> Execute 接口
class CtrlExReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val iter   = UInt(10.W)
  val is_acc = Bool()
}

// Control -> Store 接口
class CtrlStReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val wr_bank      = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val wr_bank_addr = UInt(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W)
  val iter         = UInt(10.W)
  val is_acc       = Bool()
}

// Load -> Execute 接口
class LdExReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val data   = Vec(b.veclane, UInt(b.inputType.getWidth.W))
  val iter   = UInt(10.W)
  val is_acc = Bool()
}

// Execute -> Store 接口
class ExStReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val data   = Vec(b.veclane, UInt(b.accType.getWidth.W))
  val iter   = UInt(10.W)
  val is_acc = Bool()
}
