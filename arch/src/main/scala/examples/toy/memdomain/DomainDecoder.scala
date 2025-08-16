package examples.toy.memdomain

import chisel3._
import chisel3.util._
import framework.builtin.frontend.PostDecodeCmd
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.BBISA._
import framework.builtin.mem.LocalAddr
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.Parameters

// Mem域的详细解码输出
class MemDecodeCmd(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val is_load       = Bool()
  val is_store      = Bool()
  
  // 内存地址
  val mem_addr      = UInt(bbconfig.memAddrLen.W)
  
  // 迭代次数
  val iter          = UInt(10.W)
  
  // Scratchpad地址和bank信息
  val sp_bank       = UInt(log2Up(bbconfig.sp_banks + bbconfig.acc_banks).W)
  val sp_bank_addr  = UInt(log2Up(bbconfig.spad_bank_entries + bbconfig.acc_bank_entries).W)

  // 流水线控制
  val pid           = UInt(8.W)   // 流水线ID
  val pstart        = Bool()      // 流水线的开始
  val pend          = Bool()      // 流水线的结束
}

// Mem域专用的BuckyBallCmd
class MemBuckyBallCmd(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val mem_decode_cmd = new MemDecodeCmd
  val rob_id = UInt(log2Up(bbconfig.rob_entries).W)
}

// LS decode fields
object LSDecodeFields extends Enumeration {
  type Field = Value
  val PID, PSTART, PEND, // PID大于1表示是流水线指令 
      LD_EN, ST_EN, MEMADDR, SPADDR, ITER = Value
}

// Default constants for Mem decoder
object MemDefaultConstants {
  val Y = true.B
  val N = false.B
  val DADDR = 0.U(14.W)
  val DITER = 0.U(10.W)
}

class MemDomainDecoder(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Module {
  import MemDefaultConstants._
  
  val io = IO(new Bundle {
    val post_decode_cmd_i = Flipped(Decoupled(new PostDecodeCmd))
    val mem_decode_cmd_o = Decoupled(new MemDecodeCmd)
  })

  val spAddrLen = bbconfig.spAddrLen
  val memAddrLen = bbconfig.memAddrLen

  // 只处理Mem指令
  io.post_decode_cmd_i.ready := io.mem_decode_cmd_o.ready && io.post_decode_cmd_i.bits.is_mem

  val func7 = io.post_decode_cmd_i.bits.raw_cmd.inst.funct
  val rs1   = io.post_decode_cmd_i.bits.raw_cmd.rs1
  val rs2   = io.post_decode_cmd_i.bits.raw_cmd.rs2

  // Load/Store指令解码
  import LSDecodeFields._
  val ls_default_decode = List(N,N,N,N,N,DADDR,DADDR,DITER)
  val ls_decode_list = ListLookup(func7, ls_default_decode, Array(
    MVIN_BITPAT  -> List(N,N,N,Y,N,rs1(memAddrLen-1,0),rs2(spAddrLen-1,0),rs2(spAddrLen+9,spAddrLen)), // mvin
    MVOUT_BITPAT -> List(N,N,N,N,Y,rs1(memAddrLen-1,0),rs2(spAddrLen-1,0),rs2(spAddrLen+9,spAddrLen))  // mvout
  ))

  // 输出赋值
  io.mem_decode_cmd_o.valid := io.post_decode_cmd_i.valid && io.post_decode_cmd_i.bits.is_mem
  
  io.mem_decode_cmd_o.bits.is_load := ls_decode_list(3).asBool
  io.mem_decode_cmd_o.bits.is_store := ls_decode_list(4).asBool
  io.mem_decode_cmd_o.bits.mem_addr := ls_decode_list(5).asUInt
  io.mem_decode_cmd_o.bits.iter := ls_decode_list(7).asUInt
  io.mem_decode_cmd_o.bits.pid := ls_decode_list(0).asUInt
  io.mem_decode_cmd_o.bits.pstart := ls_decode_list(1).asBool
  io.mem_decode_cmd_o.bits.pend := ls_decode_list(2).asBool

  // 地址解析
  val ls_spaddr = ls_decode_list(6).asUInt
  val ls_laddr = LocalAddr.cast_to_sp_addr(bbconfig.local_addr_t, ls_spaddr)
  
  io.mem_decode_cmd_o.bits.sp_bank := ls_laddr.mem_bank()
  io.mem_decode_cmd_o.bits.sp_bank_addr := ls_laddr.mem_row()
}