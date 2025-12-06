package framework.memdomain

import chisel3._
import chisel3.util._
import framework.frontend.decoder.PostGDCmd
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.memdomain.DISA._
import framework.memdomain.dma.LocalAddr
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.Parameters

// Detailed decode output for Mem domain
class MemDecodeCmd(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val is_load       = Bool()
  val is_store      = Bool()

  // Memory address
  val mem_addr      = UInt(b.memAddrLen.W)

  // Iteration count
  val iter          = UInt(10.W)

  // Scratchpad address and bank information
  // 3 bits, supports 8 banks (SPAD+ACC)
  val sp_bank       = UInt(log2Up(b.sp_banks + b.acc_banks).W)
  // 12 bits, uses SPAD row count (sufficient to accommodate ACC's 10-bit address)
  val sp_bank_addr  = UInt(log2Up(b.spad_bank_entries).W)

  val special       = UInt(40.W)
}


// LS decode fields
object LSDecodeFields extends Enumeration {
  type Field = Value
  val LD_EN, ST_EN, MEMADDR, SPADDR, ITER, SPECIAL, VALID = Value
}

// Default constants for Mem decoder
object MemDefaultConstants {
  val Y = true.B
  val N = false.B
  val DADDR = 0.U(14.W)
  val DITER = 0.U(10.W)
  val DSPECIAL = 0.U(40.W)
}

class MemDomainDecoder(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  import MemDefaultConstants._

  val io = IO(new Bundle {
    val raw_cmd_i = Flipped(Decoupled(new PostGDCmd))
    val mem_decode_cmd_o = Decoupled(new MemDecodeCmd)
  })

  val spAddrLen = b.spAddrLen
  val memAddrLen = b.memAddrLen

  // Only process Mem instructions
  io.raw_cmd_i.ready := io.mem_decode_cmd_o.ready

  val func7 = io.raw_cmd_i.bits.raw_cmd.inst.funct
  val rs1   = io.raw_cmd_i.bits.raw_cmd.rs1
  val rs2   = io.raw_cmd_i.bits.raw_cmd.rs2

  // Load/Store instruction decoding
  import LSDecodeFields._
  val ls_default_decode = List(N,N,DADDR,DADDR,DITER,DSPECIAL,N)
  val ls_decode_list = ListLookup(func7, ls_default_decode, Array(
    MVIN_BITPAT        -> List(Y,N,rs1(memAddrLen-1,0),rs2(spAddrLen-1,0),rs2(spAddrLen+9,spAddrLen),rs2(63,spAddrLen + 10),Y), // mvin
    MVOUT_BITPAT       -> List(N,Y,rs1(memAddrLen-1,0),rs2(spAddrLen-1,0),rs2(spAddrLen+9,spAddrLen),rs2(63,spAddrLen + 10),Y)  // mvout
  ))

  assert(!(io.raw_cmd_i.fire && !ls_decode_list(LSDecodeFields.VALID.id).asBool),
    s"MemDomainDecoder: Invalid command opcode, func7 = 0x%x\n", func7)

// -----------------------------------------------------------------------------
// Output assignment
// -----------------------------------------------------------------------------
  io.mem_decode_cmd_o.valid := io.raw_cmd_i.valid && io.raw_cmd_i.bits.is_mem

  io.mem_decode_cmd_o.bits.is_load  := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.LD_EN.id).asBool, false.B)
  io.mem_decode_cmd_o.bits.is_store := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.ST_EN.id).asBool, false.B)
  io.mem_decode_cmd_o.bits.mem_addr := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.MEMADDR.id).asUInt, 0.U(b.memAddrLen.W))
  io.mem_decode_cmd_o.bits.iter     := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.ITER.id).asUInt, 0.U(10.W))


  // Address parsing
  val ls_spaddr = ls_decode_list(LSDecodeFields.SPADDR.id).asUInt
  val ls_laddr = LocalAddr.cast_to_sp_addr(b.local_addr_t, ls_spaddr)

  io.mem_decode_cmd_o.bits.sp_bank       := Mux(io.mem_decode_cmd_o.valid, ls_laddr.mem_bank(), 0.U(log2Up(b.sp_banks + b.acc_banks).W))
  io.mem_decode_cmd_o.bits.sp_bank_addr  := Mux(io.mem_decode_cmd_o.valid, ls_laddr.mem_row(), 0.U(log2Up(b.spad_bank_entries + b.acc_bank_entries).W))
  io.mem_decode_cmd_o.bits.special       := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.SPECIAL.id).asUInt, 0.U(40.W))
}
