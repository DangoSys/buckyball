package prototype.nagisa.layernorm

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

// Control -> Load interface
class LNCtrlLdReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val op1_bank      = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val op1_bank_addr = UInt(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W)
  val iter          = UInt(10.W)
  val is_acc        = Bool()
  val norm_dim      = UInt(12.W)
  val param_bank    = UInt(2.W)
  val gamma_addr    = UInt(12.W)
  val beta_addr     = UInt(12.W)
  val use_affine    = Bool()
}

// Control -> Reduce interface
class LNCtrlReduceReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val iter       = UInt(10.W)
  val is_acc     = Bool()
  val norm_dim   = UInt(12.W)
  val use_affine = Bool()
}

// Control -> Normalize interface
class LNCtrlNormReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val iter       = UInt(10.W)
  val is_acc     = Bool()
  val norm_dim   = UInt(12.W)
  val use_affine = Bool()
}

// Control -> Store interface
class LNCtrlStReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val wr_bank      = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val wr_bank_addr = UInt(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W)
  val iter         = UInt(10.W)
  val is_acc       = Bool()
  val norm_dim     = UInt(12.W)
}

// Load -> Reduce interface: raw data with batch info
class LNLdReduceReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val data       = Vec(b.veclane, SInt(32.W))  // Always use INT32 internally
  val batch_idx  = UInt(10.W)
  val vec_idx    = UInt(12.W)
  val is_last    = Bool()
}

// Reduce -> Normalize interface: mean and rsqrt
class LNReduceNormReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val mean       = SInt(32.W)  // Q16.16 fixed point
  val rsqrt      = SInt(32.W)  // Q16.16 fixed point
  val batch_idx  = UInt(10.W)
}

// Load -> Normalize interface: gamma and beta parameters
class LNLdNormParam(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val gamma = Vec(b.veclane, SInt(32.W))
  val beta  = Vec(b.veclane, SInt(32.W))
  val vec_idx = UInt(12.W)
}

// Normalize -> Store interface: normalized data
class LNNormStReq(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val data       = Vec(b.veclane, SInt(32.W))
  val batch_idx  = UInt(10.W)
  val vec_idx    = UInt(12.W)
  val is_last    = Bool()
}

// Special field decoder
class LayerNormSpecial extends Bundle {
  val norm_dim    = UInt(12.W)
  val gamma_addr  = UInt(12.W)
  val beta_addr   = UInt(12.W)
  val param_bank  = UInt(2.W)
  val use_affine  = Bool()
  val reserved    = Bool()
}

object LayerNormSpecial {
  def decode(special: UInt): LayerNormSpecial = {
    val s = Wire(new LayerNormSpecial)
    s.norm_dim    := special(11, 0)
    s.gamma_addr  := special(23, 12)
    s.beta_addr   := special(35, 24)
    s.param_bank  := special(37, 36)
    s.use_affine  := special(38)
    s.reserved    := special(39)
    s
  }
}
