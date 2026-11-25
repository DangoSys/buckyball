package examples.balls.gemmini

import framework.generic.contract.ball.BallHostParam

import examples.balls.gemmini.configs.GemminiBallParam

import chisel3._
import chisel3.util._

object LoopSubCmdType {
  val MSET_ALLOC = 0.U(3.W)
  val MSET_FREE  = 1.U(3.W)
  val MVIN       = 2.U(3.W)
  val MVOUT      = 3.U(3.W)
  val PRELOAD    = 4.U(3.W)
  val COMPUTE    = 5.U(3.W)
}

class LoopSubCmd(host: BallHostParam, val ballParam: GemminiBallParam) extends Bundle {
  val cmdType       = UInt(3.W)
  val bank_id       = UInt(host.link.bankIdBits.W)
  val dram_addr     = UInt(host.memAddrBits.W)
  val stride        = UInt(19.W)
  val iter          = UInt(host.link.iterBits.W)
  val bank_row      = UInt(5.W)
  val bank_col      = UInt(5.W)
  val op1_bank      = UInt(host.link.bankIdBits.W)
  val op2_bank      = UInt(host.link.bankIdBits.W)
  val wr_bank       = UInt(host.link.bankIdBits.W)
  val compute_mode  = UInt(2.W)
  val zero_op2      = Bool()
  val zero_op1_tail = Bool()
}

class LoopCmd(host: BallHostParam, val ballParam: GemminiBallParam) extends Bundle {
  val slots = Vec(4, Valid(new LoopSubCmd(host, ballParam)))
  val last  = Bool()
}

class LoopWsConfig(host: BallHostParam, val ballParam: GemminiBallParam) extends Bundle {
  val max_i       = UInt(16.W)
  val max_j       = UInt(16.W)
  val max_k       = UInt(16.W)
  val dram_addr_a = UInt(host.memAddrBits.W)
  val dram_addr_b = UInt(host.memAddrBits.W)
  val dram_addr_d = UInt(host.memAddrBits.W)
  val dram_addr_c = UInt(host.memAddrBits.W)
  val stride_a    = UInt(32.W)
  val stride_b    = UInt(32.W)
  val stride_d    = UInt(32.W)
  val stride_c    = UInt(32.W)
  val bank_a      = UInt(host.link.bankIdBits.W)
  val bank_b      = UInt(host.link.bankIdBits.W)
  val bank_c      = UInt(host.link.bankIdBits.W)
  val low_d       = Bool()
}

class LoopConvWsConfig(host: BallHostParam, val ballParam: GemminiBallParam) extends Bundle {
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
  val dram_addr_bias   = UInt(host.memAddrBits.W)
  val dram_addr_input  = UInt(host.memAddrBits.W)
  val dram_addr_weight = UInt(host.memAddrBits.W)
  val dram_addr_output = UInt(host.memAddrBits.W)
  val input_stride     = UInt(32.W)
  val weight_stride    = UInt(32.W)
  val output_stride    = UInt(32.W)
  val bank_input       = UInt(host.link.bankIdBits.W)
  val bank_weight      = UInt(host.link.bankIdBits.W)
  val bank_output      = UInt(host.link.bankIdBits.W)
  val no_bias          = Bool()
}
