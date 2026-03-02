package framework.memdomain.frontend.cmd_channel.decoder

import chisel3._
import chisel3.util._
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd_channel.decoder.DISA._
import freechips.rocketchip.tile._
import chisel3.experimental.hierarchy.{instantiable, public}

// Detailed decode output for Mem domain
class MemDecodeCmd(b: GlobalConfig) extends Bundle {
  val is_load   = Bool()
  val is_store  = Bool()
  val is_config = Bool()

  // Memory address
  val mem_addr = UInt(b.memDomain.memAddrLen.W)
  // Iteration count
  val iter     = UInt(10.W)
  // Bank information
  // 3 bits, supports 8 banks (SPAD+ACC)
  val bank_id  = UInt(log2Up(b.memDomain.bankNum).W)
  val special  = UInt(40.W)
}

// LS decode fields
object LSDecodeFields extends Enumeration {
  type Field = Value
  val LD_EN, ST_EN, MEMADDR, BANK_ID, ITER, SPECIAL, VALID = Value
}

// Default constants for Mem decoder
object MemDefaultConstants {
  val Y        = true.B
  val N        = false.B
  val DADDR    = 0.U(14.W)
  val DITER    = 0.U(10.W)
  val DSPECIAL = 0.U(40.W)
}

@instantiable
class MemDomainDecoder(val b: GlobalConfig) extends Module {
  import MemDefaultConstants._

  @public
  val io = IO(new Bundle {
    val cmd_i            = Flipped(Decoupled(new PostGDCmd(b)))
    val mem_decode_cmd_o = Decoupled(new MemDecodeCmd(b))
  })

  val bankAddrLen = log2Up(b.memDomain.bankEntries)
  val memAddrLen  = b.memDomain.memAddrLen
  val bankIdLen   = b.frontend.bank_id_len

  // Only process Mem instructions
  io.cmd_i.ready := io.mem_decode_cmd_o.ready

  val func7 = io.cmd_i.bits.cmd.funct
  val rs1   = io.cmd_i.bits.cmd.rs1Data
  val rs2   = io.cmd_i.bits.cmd.rs2Data

  // Load/Store instruction decoding
  // New unified encoding:
  //   rs1[7:0]       = bank_id
  //   rs1[26:24]     = bank access valid flags (used by scoreboard, ignored here)
  //   rs1[63:27]     = mem_addr (for MVIN/MVOUT)
  //   rs2[9:0]       = iter
  //   rs2[63:10]     = special
  import LSDecodeFields._
  val ls_default_decode = List(N, N, DADDR, DADDR, DITER, DSPECIAL, N)

  val ls_decode_list = ListLookup(
    func7,
    ls_default_decode,
    Array(
      MSET_BITPAT  -> List(
        N,
        N,
        0.U(memAddrLen.W),        // mem_addr: not used for MSET
        rs1(bankIdLen - 1, 0),    // bank_id from rs1[bankIdLen-1:0]
        rs2(9, 0),                // iter from rs2[9:0]
        rs2(39, 0),               // special from rs2[39:0]
        Y
      ), // mset
      MVIN_BITPAT  -> List(
        Y,
        N,
        rs1(memAddrLen + 26, 27), // mem_addr from rs1 upper bits
        rs1(bankIdLen - 1, 0),    // bank_id from rs1[bankIdLen-1:0]
        rs2(9, 0),                // iter from rs2[9:0]
        rs2(63, 10),              // special from rs2[63:10]
        Y
      ), // mvin
      MVOUT_BITPAT -> List(
        N,
        Y,
        rs1(memAddrLen + 26, 27), // mem_addr from rs1 upper bits
        rs1(bankIdLen - 1, 0),    // bank_id from rs1[bankIdLen-1:0]
        rs2(9, 0),                // iter from rs2[9:0]
        rs2(63, 10),              // special from rs2[63:10]
        Y
      )  // mvout
    )
  )

  assert(
    !(io.cmd_i.fire && !ls_decode_list(LSDecodeFields.VALID.id).asBool),
    s"MemDomainDecoder: Invalid command opcode, func7 = 0x%x\n",
    func7
  )

// -----------------------------------------------------------------------------
// Output assignment
// -----------------------------------------------------------------------------
  io.mem_decode_cmd_o.valid := io.cmd_i.valid && (io.cmd_i.bits.domain_id === DomainId.MEM)

  io.mem_decode_cmd_o.bits.is_load   := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.LD_EN.id).asBool,
    false.B
  )
  io.mem_decode_cmd_o.bits.is_store  := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.ST_EN.id).asBool,
    false.B
  )
  io.mem_decode_cmd_o.bits.is_config := Mux(
    io.mem_decode_cmd_o.valid,
    func7 === MSET_BITPAT,
    false.B
  )
  io.mem_decode_cmd_o.bits.mem_addr  := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.MEMADDR.id).asUInt,
    0.U(b.memDomain.memAddrLen.W)
  )
  io.mem_decode_cmd_o.bits.iter      := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.ITER.id).asUInt,
    0.U(10.W)
  )

  // Address parsing
  val ls_bank_id = ls_decode_list(LSDecodeFields.BANK_ID.id).asUInt
  io.mem_decode_cmd_o.bits.bank_id := Mux(io.mem_decode_cmd_o.valid, ls_bank_id, 0.U(log2Up(b.memDomain.bankNum).W))
  io.mem_decode_cmd_o.bits.special := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.SPECIAL.id).asUInt,
    0.U(40.W)
  )
}
