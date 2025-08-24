package examples.toy.balldomain

import chisel3._
import chisel3.util._
import framework.builtin.frontend.PostDecodeCmd
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain.DISA._
import framework.builtin.memdomain.dma.LocalAddr
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.Parameters

// EX域的详细解码输出
class ExDecodeCmd(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val is_matmul_ws  = Bool()
  val is_vec        = Bool()
  val is_bbfp       = Bool()
  val is_fence      = Bool()
  
  // 迭代次数
  val iter          = UInt(10.W)
  
  // Execute专用字段
  val op1_en        = Bool()
  val op2_en        = Bool()
  val wr_spad_en    = Bool()
  val op1_from_spad = Bool()
  val op2_from_spad = Bool()
  
  // Execute的操作数地址
  val op1_bank      = UInt(log2Up(b.sp_banks).W)
  val op1_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  val op2_bank      = UInt(log2Up(b.sp_banks).W)
  val op2_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  
  // 写入地址和bank信息
  val wr_bank       = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val wr_bank_addr  = UInt(log2Up(b.spad_bank_entries + b.acc_bank_entries).W)
  val is_acc        = Bool() // 是否是acc bank的操作    
}

// EX域专用的BuckyBallCmd
class ExBuckyBallCmd(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ex_decode_cmd = new ExDecodeCmd
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// EX decode fields
object EXDecodeFields extends Enumeration {
  type Field = Value
  val OP1_EN, OP2_EN, WR_SPAD, OP1_FROM_SPAD, OP2_FROM_SPAD, FENCE_EN, OP1_SPADDR, OP2_SPADDR, WR_SPADDR, ITER = Value
}

// Default constants for EX decoder
object ExDefaultConstants {
  val Y = true.B
  val N = false.B
  val DADDR = 0.U(14.W)
  val DITER = 0.U(10.W)
}

class ExDomainDecoder(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  import ExDefaultConstants._
  
  val io = IO(new Bundle {
    val post_decode_cmd_i = Flipped(Decoupled(new PostDecodeCmd))
    val ex_decode_cmd_o = Decoupled(new ExDecodeCmd)
  })

  val spAddrLen = b.spAddrLen

  // 只处理EX指令
  io.post_decode_cmd_i.ready := io.ex_decode_cmd_o.ready && io.post_decode_cmd_i.bits.is_ex

  val func7 = io.post_decode_cmd_i.bits.raw_cmd.inst.funct
  val rs1   = io.post_decode_cmd_i.bits.raw_cmd.rs1
  val rs2   = io.post_decode_cmd_i.bits.raw_cmd.rs2

  // EX指令解码
  import EXDecodeFields._
  val ex_default_decode = List(N,N,N,N,N,N,N,N,N,DADDR,DADDR,DADDR,DITER)
  val ex_decode_list = ListLookup(func7, ex_default_decode, Array(
    MATMUL_WARP16_BITPAT -> List(N,N,N,Y,Y,Y,Y,Y,N,rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen)),
    BB_BBFP_MUL          -> List(N,N,N,Y,Y,Y,Y,Y,N,rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen)),
    MATMUL_WS            -> List(N,N,N,Y,Y,Y,Y,Y,N,rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen)),
    FENCE                -> List(N,N,N,N,N,N,N,N,Y,             DADDR,                          DADDR,              DADDR,                    1.U(10.W))
  ))

  // 输出赋值
  io.ex_decode_cmd_o.valid := io.post_decode_cmd_i.valid && io.post_decode_cmd_i.bits.is_ex
  
  io.ex_decode_cmd_o.bits.is_fence      := ex_decode_list(EXDecodeFields.FENCE_EN.id).asBool
  io.ex_decode_cmd_o.bits.is_vec        := func7 === MATMUL_WARP16_BITPAT
  io.ex_decode_cmd_o.bits.is_bbfp       := func7 === BB_BBFP_MUL || func7 === MATMUL_WS
  io.ex_decode_cmd_o.bits.is_matmul_ws  := func7 === MATMUL_WS
  
  io.ex_decode_cmd_o.bits.iter          := ex_decode_list(EXDecodeFields.ITER.id).asUInt

  io.ex_decode_cmd_o.bits.op1_en        := ex_decode_list(EXDecodeFields.OP1_EN.id).asBool
  io.ex_decode_cmd_o.bits.op2_en        := ex_decode_list(EXDecodeFields.OP2_EN.id).asBool
  io.ex_decode_cmd_o.bits.wr_spad_en    := ex_decode_list(EXDecodeFields.WR_SPAD.id).asBool
  io.ex_decode_cmd_o.bits.op1_from_spad := ex_decode_list(EXDecodeFields.OP1_FROM_SPAD.id).asBool
  io.ex_decode_cmd_o.bits.op2_from_spad := ex_decode_list(EXDecodeFields.OP2_FROM_SPAD.id).asBool

  // 地址解析
  val op1_spaddr = ex_decode_list(EXDecodeFields.OP1_SPADDR.id).asUInt
  val op2_spaddr = ex_decode_list(EXDecodeFields.OP2_SPADDR.id).asUInt  
  val wr_spaddr = ex_decode_list(EXDecodeFields.WR_SPADDR.id).asUInt
  
  val op1_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, op1_spaddr)
  val op2_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, op2_spaddr)
  val wr_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, wr_spaddr)
  
  io.ex_decode_cmd_o.bits.op1_bank := op1_laddr.sp_bank()
  io.ex_decode_cmd_o.bits.op1_bank_addr := op1_laddr.sp_row()
  io.ex_decode_cmd_o.bits.op2_bank := op2_laddr.sp_bank()
  io.ex_decode_cmd_o.bits.op2_bank_addr := op2_laddr.sp_row()
  
  io.ex_decode_cmd_o.bits.wr_bank := wr_laddr.mem_bank()
  io.ex_decode_cmd_o.bits.wr_bank_addr := wr_laddr.mem_row()
  io.ex_decode_cmd_o.bits.is_acc := (io.ex_decode_cmd_o.bits.wr_bank >= b.sp_banks.U)
  
  // 断言：执行指令中OpA和OpB必须访问不同的bank
  assert(!(io.ex_decode_cmd_o.valid && io.ex_decode_cmd_o.bits.op1_en && io.ex_decode_cmd_o.bits.op2_en && 
           io.ex_decode_cmd_o.bits.op1_bank === io.ex_decode_cmd_o.bits.op2_bank), 
    "ExDomainDecoder: Execute instruction OpA and OpB cannot access the same bank")
}