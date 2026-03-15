package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig

object LoopSubCmdType {
  val MSET_ALLOC = 0.U(3.W)
  val MSET_FREE  = 1.U(3.W)
  val MVIN       = 2.U(3.W)
  val MVOUT      = 3.U(3.W)
  val PRELOAD    = 4.U(3.W)
  val COMPUTE    = 5.U(3.W)
}

class LoopSubCmd(val b: GlobalConfig) extends Bundle {
  val cmdType      = UInt(3.W)
  val bank_id      = UInt(log2Up(b.memDomain.bankNum).W)
  val dram_addr    = UInt(b.memDomain.memAddrLen.W)
  val iter         = UInt(b.frontend.iter_len.W)
  val bank_row     = UInt(5.W)
  val bank_col     = UInt(5.W)
  val op1_bank     = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_bank     = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_bank      = UInt(log2Up(b.memDomain.bankNum).W)
  val compute_mode = UInt(2.W)
}

class LoopCmd(val b: GlobalConfig) extends Bundle {
  val slots = Vec(4, Valid(new LoopSubCmd(b)))
}

class LoopWsConfig(val b: GlobalConfig) extends Bundle {
  val max_i       = UInt(16.W)
  val max_j       = UInt(16.W)
  val max_k       = UInt(16.W)
  val dram_addr_a = UInt(b.memDomain.memAddrLen.W)
  val dram_addr_b = UInt(b.memDomain.memAddrLen.W)
  val dram_addr_d = UInt(b.memDomain.memAddrLen.W)
  val dram_addr_c = UInt(b.memDomain.memAddrLen.W)
  val stride_a    = UInt(32.W)
  val stride_b    = UInt(32.W)
  val stride_d    = UInt(32.W)
  val stride_c    = UInt(32.W)
  val bank_a      = UInt(log2Up(b.memDomain.bankNum).W)
  val bank_b      = UInt(log2Up(b.memDomain.bankNum).W)
  val bank_c      = UInt(log2Up(b.memDomain.bankNum).W)
  val low_d       = Bool()
}

class LoopConvWsConfig(val b: GlobalConfig) extends Bundle {
  val batch_size       = UInt(16.W)
  val in_dim           = UInt(16.W)
  val in_channels      = UInt(16.W)
  val out_channels     = UInt(16.W)
  val out_dim          = UInt(16.W)
  val stride           = UInt(8.W)
  val padding          = UInt(8.W)
  val kernel_dim       = UInt(8.W)
  val pool_size        = UInt(8.W)
  val pool_stride      = UInt(8.W)
  val pool_padding     = UInt(8.W)
  val dram_addr_bias   = UInt(b.memDomain.memAddrLen.W)
  val dram_addr_input  = UInt(b.memDomain.memAddrLen.W)
  val dram_addr_weight = UInt(b.memDomain.memAddrLen.W)
  val dram_addr_output = UInt(b.memDomain.memAddrLen.W)
  val input_stride     = UInt(32.W)
  val weight_stride    = UInt(32.W)
  val output_stride    = UInt(32.W)
  val bank_input       = UInt(log2Up(b.memDomain.bankNum).W)
  val bank_weight      = UInt(log2Up(b.memDomain.bankNum).W)
  val bank_output      = UInt(log2Up(b.memDomain.bankNum).W)
  val no_bias          = Bool()
}
