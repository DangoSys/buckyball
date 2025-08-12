package buckyball.frontend

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters
import buckyball.BuckyBallConfig
// import framework.ballcore.ballcore.RoCCCommandBB
import freechips.rocketchip.tile._  
import buckyball.BBISA._
import buckyball.mem.LocalAddr


class BuckyBallRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommand
}

// default values
object DefaultConstants {
  val Y = true.B
  val N = false.B

  val DOP      = VecInit(Seq.fill(16)(0.U(8.W)))
  val DADDR    = 0.U(14.W)
  val DVECIDX  = 0.U(5.W)
  val DITER    = 0.U(10.W)
  val DTC_TYPE = 0.U(2.W)
  val ZERO_VEC = VecInit(Seq.fill(16)(0.U(8.W)))
}

object LSDecodeFields extends Enumeration {
  type Field = Value
  val PID, PSTART, PEND, // PID大于1表示是流水线指令 
      LD_EN, ST_EN, MEMADDR, SPADDR, ITER = Value
}

// index of the decoded fields
object EXDecodeFields extends Enumeration {
  type Field = Value
  val PID, PSTART, PEND, // PID大于1表示是流水线指令 
      OP1_EN, OP2_EN, WR_SPAD, OP1_FROM_SPAD, OP2_FROM_SPAD, OP1_SPADDR, OP2_SPADDR, WR_SPADDR,
      ITER = Value
}
object FENCEDecodeFields extends Enumeration {
  type Field = Value
  val PID, PSTART, PEND,
      FN_EN = Value
}

class PostDecodeCmd(implicit bbconfig: BuckyBallConfig) extends Bundle {
  val is_matmul_ws  = Bool()
  val is_load       = Bool()
  val is_store      = Bool()
  val is_ex         = Bool()
  val is_vec        = Bool()
  val is_bbfp       = Bool()
  val is_fence      = Bool() 
  // 内存地址 - 只用于load/store
  val mem_addr      = UInt(bbconfig.memAddrLen.W)
  
  // 迭代次数 - 所有指令都可能用到
  val iter          = UInt(10.W)
  
  // Scratchpad读取地址和bank信息 - store源地址
  val rd_bank       = UInt(log2Up(bbconfig.sp_banks+ bbconfig.acc_banks).W)
  val rd_bank_addr  = UInt(log2Up(bbconfig.spad_bank_entries+ bbconfig.acc_bank_entries).W)
  
  // Scratchpad写入地址和bank信息 - load目标地址，execute结果地址(后续拆到acc中)
  val wr_bank       = UInt(log2Up(bbconfig.sp_banks + bbconfig.acc_banks).W)
  val wr_bank_addr  = UInt(log2Up(bbconfig.spad_bank_entries + bbconfig.acc_bank_entries).W)
  val is_acc        = Bool() // 是否是acc bank的操作    
  
  // Execute专用字段
  val op1_en        = Bool()
  val op2_en        = Bool()
  val wr_spad_en    = Bool()
  val op1_from_spad = Bool()
  val op2_from_spad = Bool()
  
  // Execute的操作数地址（保留原始字段名）
  val op1_bank      = UInt(log2Up(bbconfig.sp_banks).W)
  val op1_bank_addr = UInt(log2Up(bbconfig.spad_bank_entries).W)
  val op2_bank      = UInt(log2Up(bbconfig.sp_banks).W)
  val op2_bank_addr = UInt(log2Up(bbconfig.spad_bank_entries).W)

  // 流水线控制
  val pid           = UInt(8.W)   // 流水线ID
  val pstart        = Bool()      // 流水线的开始
  val pend          = Bool()      // 流水线的结束
}

class Decoder(implicit bbconfig: BuckyBallConfig, p: Parameters) extends Module {
  import DefaultConstants._

  val io = IO(new Bundle {
    val id_i = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommand
    }))
    val id_rs = Decoupled(new PostDecodeCmd)
  })

  val spAddrLen = bbconfig.spAddrLen
  val memAddrLen = bbconfig.memAddrLen

  io.id_i.ready := io.id_rs.ready // 如果保留站阻塞了，id_i也阻塞

  val func7 = io.id_i.bits.cmd.inst.funct
  val rs1   = io.id_i.bits.cmd.rs1
  val rs2   = io.id_i.bits.cmd.rs2

// -----------------------------------------------------------------------------
// Decode Load/Store instructions
// -----------------------------------------------------------------------------
  import LSDecodeFields._
  val ls_default_decode = List(N,N,N,N,N,DADDR,DADDR,DITER)
  val ls_decode_list = ListLookup(func7, ls_default_decode, Array(
    MVIN_BITPAT  -> List(N,N,N,Y,N,rs1(memAddrLen-1,0),rs2(spAddrLen-1,0),rs2(spAddrLen+9,spAddrLen)), // mvin
    MVOUT_BITPAT -> List(N,N,N,N,Y,rs1(memAddrLen-1,0),rs2(spAddrLen-1,0),rs2(spAddrLen+9,spAddrLen)), // mvout
  ))

// -----------------------------------------------------------------------------
// Decode EX instructions
// -----------------------------------------------------------------------------
  import EXDecodeFields._
  val ex_default_decode = List(N,N,N,N,N,N,N,N,DADDR,DADDR,DADDR,DITER)
  val ex_decode_list = ListLookup(func7, ex_default_decode, Array(
    MATMUL_WARP16_BITPAT -> List(N,N,N,Y,Y,Y,Y,Y,rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen)), // bb_matmul_warp16
    BB_BBFP_MUL          -> List(N,N,N,Y,Y,Y,Y,Y,rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen)), // bb_bbfp_mul
    MATMUL_WS            -> List(N,N,N,Y,Y,Y,Y,Y,rs1(spAddrLen-1,0), rs1(2*spAddrLen - 1,spAddrLen), rs2(spAddrLen-1,0), rs2(spAddrLen + 9,spAddrLen)), // matmul_ws
  ))
// -----------------------------------------------------------------------------
// Fence instructions
// -----------------------------------------------------------------------------
  import FENCEDecodeFields._
  val fence_default_decode = List(N,N,N,N)
  val fence_decode_list = ListLookup(func7, fence_default_decode, Array(
    FENCE  -> List(N,N,N,Y)
  ))

  io.id_rs.valid              := io.id_i.valid
  io.id_rs.bits.is_load       := ls_decode_list(3).asBool
  io.id_rs.bits.is_store      := ls_decode_list(4).asBool
  io.id_rs.bits.mem_addr      := ls_decode_list(5).asUInt
  io.id_rs.bits.is_ex         := !ls_decode_list(3).asBool && !ls_decode_list(4).asBool && !fence_decode_list(3).asBool
  io.id_rs.bits.is_fence      := fence_decode_list(3).asBool
  io.id_rs.bits.iter          := Mux(io.id_rs.bits.is_ex, ex_decode_list(11).asUInt, 
                                  Mux(io.id_rs.bits.is_load || io.id_rs.bits.is_store, ls_decode_list(7).asUInt, 0.U))

  io.id_rs.bits.pid           := Mux(io.id_rs.bits.is_ex, ex_decode_list(0).asUInt, 
                                  Mux(io.id_rs.bits.is_load, ls_decode_list(0).asUInt, 0.U))
  io.id_rs.bits.pstart        := ex_decode_list(1).asBool || ls_decode_list(1).asBool 
  io.id_rs.bits.pend          := ex_decode_list(2).asBool || ls_decode_list(2).asBool

  io.id_rs.bits.op1_en        := ex_decode_list(3).asBool && io.id_i.valid
  io.id_rs.bits.op2_en        := ex_decode_list(4).asBool && io.id_i.valid
  io.id_rs.bits.wr_spad_en    := ex_decode_list(5).asBool && io.id_i.valid
  io.id_rs.bits.op1_from_spad := ex_decode_list(6).asBool
  io.id_rs.bits.op2_from_spad := ex_decode_list(7).asBool
  io.id_rs.bits.is_vec        := func7 === MATMUL_WARP16_BITPAT
  io.id_rs.bits.is_bbfp       := func7 === BB_BBFP_MUL||func7 === MATMUL_WS
  io.id_rs.bits.is_matmul_ws  := func7 === MATMUL_WS
  // LocalAddr解析 - 在解码阶段完成bank和本地地址的计算
  // 地址映射 (4个bank，每个bank 4096行)：
  // spaddr[13:12] -> bank_num (0-3)
  // spaddr[11:0]  -> bank_addr (0-4095)
  
  // 从原始指令字段解析地址
  val op1_spaddr = ex_decode_list(8).asUInt  // rs1[spAddrLen-1:0]
  val op2_spaddr = ex_decode_list(9).asUInt  // rs1[2*spAddrLen-1:spAddrLen] 
  val wr_spaddr = ex_decode_list(10).asUInt  // rs2[spAddrLen-1:0]
  val ls_spaddr = ls_decode_list(6).asUInt   // load/store的sp_addr
  
  val op1_laddr = LocalAddr.cast_to_sp_addr(bbconfig.local_addr_t, op1_spaddr)
  val op2_laddr = LocalAddr.cast_to_sp_addr(bbconfig.local_addr_t, op2_spaddr)
  val wr_laddr = LocalAddr.cast_to_sp_addr(bbconfig.local_addr_t, wr_spaddr)
  val ls_laddr = LocalAddr.cast_to_sp_addr(bbconfig.local_addr_t, ls_spaddr)
  
  // 根据指令类型分配bank信息
  // Load: wr_bank = ls_spaddr解析的bank (写入scratchpad), rd_bank不用
  // Store: rd_bank = ls_spaddr解析的bank (从scratchpad读取), wr_bank不用  
  // Execute: rd_bank = op1_spaddr解析的bank (OpA，也作为读取), op1_bank/op2_bank = 操作数bank, wr_bank = wr_spaddr解析的bank (结果)
  
  io.id_rs.bits.rd_bank := Mux(io.id_rs.bits.is_ex, op1_laddr.mem_bank(), ls_laddr.mem_bank())
  io.id_rs.bits.rd_bank_addr := Mux(io.id_rs.bits.is_ex, op1_laddr.mem_row(), ls_laddr.mem_row())
  
  io.id_rs.bits.wr_bank := Mux(io.id_rs.bits.is_ex, wr_laddr.mem_bank(), ls_laddr.mem_bank())
  io.id_rs.bits.wr_bank_addr := Mux(io.id_rs.bits.is_ex, wr_laddr.mem_row(), ls_laddr.mem_row())
  io.id_rs.bits.is_acc := (io.id_rs.bits.wr_bank >= bbconfig.sp_banks.U) // 如果wr_bank大于sp_banks，则是acc bank操作

  io.id_rs.bits.op1_bank := op1_laddr.sp_bank()  // execute的OpA
  io.id_rs.bits.op1_bank_addr := op1_laddr.sp_row()
  io.id_rs.bits.op2_bank := op2_laddr.sp_bank()  // execute的OpB
  io.id_rs.bits.op2_bank_addr := op2_laddr.sp_row()
  
  // 断言：执行指令中OpA和OpB必须访问不同的bank
  assert(!(io.id_rs.bits.is_ex && io.id_rs.bits.op1_en && io.id_rs.bits.op2_en && 
           io.id_rs.bits.op1_bank === io.id_rs.bits.op2_bank), 
    "Decoder: Execute instruction OpA and OpB cannot access the same bank")
}


