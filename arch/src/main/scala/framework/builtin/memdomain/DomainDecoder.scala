package framework.builtin.memdomain

import chisel3._
import chisel3.util._
import framework.builtin.frontend.PostGDCmd
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.DISA._
import framework.builtin.memdomain.dma.LocalAddr
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.Parameters

// Mem域的详细解码输出
class MemDecodeCmd(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val is_load       = Bool()
  val is_store      = Bool()
  
  // 内存地址
  val mem_addr      = UInt(b.memAddrLen.W)
  
  // 迭代次数
  val iter          = UInt(10.W)
  
  // Scratchpad地址和bank信息
  val sp_bank       = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  val sp_bank_addr  = UInt(log2Up(b.spad_bank_entries + b.acc_bank_entries).W)
}


// LS decode fields
object LSDecodeFields extends Enumeration {
  type Field = Value
  val LD_EN, ST_EN, MEMADDR, SPADDR, ITER, VALID = Value
}

// Default constants for Mem decoder
object MemDefaultConstants {
  val Y = true.B
  val N = false.B
  val DADDR = 0.U(14.W)
  val DITER = 0.U(10.W)
}

class MemDomainDecoder(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  import MemDefaultConstants._
  
  val io = IO(new Bundle {
    val raw_cmd_i = Flipped(Decoupled(new PostGDCmd))
    val mem_decode_cmd_o = Decoupled(new MemDecodeCmd)
  })

  val spAddrLen = b.spAddrLen
  val memAddrLen = b.memAddrLen

  // 只处理Mem指令
  io.raw_cmd_i.ready := io.mem_decode_cmd_o.ready

  val func7 = io.raw_cmd_i.bits.raw_cmd.inst.funct
  val rs1   = io.raw_cmd_i.bits.raw_cmd.rs1
  val rs2   = io.raw_cmd_i.bits.raw_cmd.rs2

  // Load/Store指令解码
  import LSDecodeFields._
  val ls_default_decode = List(N,N,DADDR,DADDR,DITER,N)
  val ls_decode_list = ListLookup(func7, ls_default_decode, Array(
    MVIN_BITPAT        -> List(Y,N,rs1(memAddrLen-1,0),rs2(spAddrLen-1,0),rs2(spAddrLen+9,spAddrLen),Y), // mvin
    MVOUT_BITPAT       -> List(N,Y,rs1(memAddrLen-1,0),rs2(spAddrLen-1,0),rs2(spAddrLen+9,spAddrLen),Y)  // mvout
  ))

  assert(!(io.raw_cmd_i.fire && !ls_decode_list(LSDecodeFields.VALID.id).asBool), 
    s"MemDomainDecoder: Invalid command opcode, func7 = 0x%x\n", func7)

// -----------------------------------------------------------------------------
// 输出赋值
// -----------------------------------------------------------------------------
  io.mem_decode_cmd_o.valid := io.raw_cmd_i.valid && io.raw_cmd_i.bits.is_mem
  
  io.mem_decode_cmd_o.bits.is_load  := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.LD_EN.id).asBool, false.B)
  io.mem_decode_cmd_o.bits.is_store := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.ST_EN.id).asBool, false.B)
  io.mem_decode_cmd_o.bits.mem_addr := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.MEMADDR.id).asUInt, 0.U(b.memAddrLen.W))
  io.mem_decode_cmd_o.bits.iter     := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.ITER.id).asUInt, 0.U(10.W))


  // 地址解析
  val ls_spaddr = ls_decode_list(LSDecodeFields.SPADDR.id).asUInt
  val ls_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, ls_spaddr)
  
  io.mem_decode_cmd_o.bits.sp_bank       := Mux(io.mem_decode_cmd_o.valid, ls_laddr.mem_bank(), 0.U(log2Up(b.sp_banks + b.acc_banks).W))
  io.mem_decode_cmd_o.bits.sp_bank_addr  := Mux(io.mem_decode_cmd_o.valid, ls_laddr.mem_row(), 0.U(log2Up(b.spad_bank_entries + b.acc_bank_entries).W))
}