package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

// Control -> Load interface
class SMCtrlLdReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val op1_bank      = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val op1_bank_addr = UInt(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W)
  val iter          = UInt(10.W)
  val is_acc        = Bool()
  val dim_len       = UInt(10.W)
  val batch         = UInt(10.W)
  val log_mode      = Bool()
}

// Control -> FindMax interface
class SMCtrlFindMaxReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val iter       = UInt(10.W)
  val dim_len    = UInt(10.W)
  val batch      = UInt(10.W)
}

// Control -> ExpSum interface
class SMCtrlExpSumReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val iter       = UInt(10.W)
  val dim_len    = UInt(10.W)
  val batch      = UInt(10.W)
}

// Control -> Normalize interface
class SMCtrlNormReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val iter       = UInt(10.W)
  val dim_len    = UInt(10.W)
  val batch      = UInt(10.W)
  val log_mode   = Bool()
}

// Control -> Store interface
class SMCtrlStReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val wr_bank      = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val wr_bank_addr = UInt(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W)
  val iter         = UInt(10.W)
  val is_acc       = Bool()
  val dim_len      = UInt(10.W)
}

// Load -> FindMax interface: raw data
class SMLdFindMaxReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val data       = Vec(b.veclane, SInt(32.W))  // Always use INT32 internally
  val vec_idx    = UInt(12.W)
  val batch_idx  = UInt(10.W)
}

// FindMax -> ExpSum interface: max value per batch
class SMFindMaxExpSumReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val max_val    = SInt(32.W)  // FP32 max value
  val batch_idx  = UInt(10.W)
}

// ExpSum -> Normalize interface: sum of exp values
class SMExpSumNormReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val sum_exp    = UInt(32.W)  // FP32 sum of exp
  val batch_idx  = UInt(10.W)
}

// Normalize -> Store interface: normalized data
class SMNormStReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val data       = Vec(b.veclane, SInt(32.W))
  val vec_idx    = UInt(12.W)
  val batch_idx  = UInt(10.W)
  val is_last    = Bool()
}

// Special field decoder for Softmax
class SoftmaxSpecial extends Bundle {
  val dim_len  = UInt(10.W)  // Softmax dimension length (1-1024)
  val batch    = UInt(10.W)  // Batch size (1-1024)
  val log_mode = Bool()      // 0=Softmax, 1=LogSoftmax
  val reserved = UInt(19.W)  // Reserved bits
}

object SoftmaxSpecial {
  def decode(special: UInt): SoftmaxSpecial = {
    val s = Wire(new SoftmaxSpecial)
    s.dim_len  := special(9, 0)
    s.batch    := special(19, 10)
    s.log_mode := special(20)
    s.reserved := special(39, 21)
    s
  }

  def encode(dim_len: UInt, batch: UInt, log_mode: Bool): UInt = {
    Cat(0.U(19.W), log_mode, batch, dim_len)
  }
}
