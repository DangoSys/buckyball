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
  val is_shared = Bool() // Shared memory access marker derived from bank_id threshold.
  val is_load   = Bool()
  val is_store  = Bool()
  val is_config = Bool()

  // Memory address
  val mem_addr = UInt(b.memDomain.memAddrLen.W)
  // Iteration count
  val iter     = UInt(b.frontend.iter_len.W)
  // Bank information
  val bank_id  = UInt(log2Up(b.memDomain.bankNum).W)
  val special  = UInt(64.W)
}

// LS decode fields (iter removed from decode table — always from rs1[63:48])
object LSDecodeFields extends Enumeration {
  type Field = Value
  val LD_EN, ST_EN, MEMADDR, BANK_ID, SPECIAL, VALID = Value
}

// Default constants for Mem decoder
object MemDefaultConstants {
  val Y        = true.B
  val N        = false.B
  val DADDR    = 0.U(15.W)
  val DSPECIAL = 0.U(64.W)
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
  val iterLen     = b.frontend.iter_len

  // Only process Mem instructions
  io.cmd_i.ready := io.mem_decode_cmd_o.ready

  val func7 = io.cmd_i.bits.cmd.funct
  val rs1   = io.cmd_i.bits.cmd.rs1Data
  val rs2   = io.cmd_i.bits.cmd.rs2Data

  // Unified encoding:
  //   rs1[14:0]  = bank_id (BANK0)
  //   rs1[45:47] = bank access valid flags (used by scoreboard, ignored here)
  //   rs1[63:48] = iter
  //   rs2[38:0]  = mem_addr (for MVIN/MVOUT, 39-bit)
  //   rs2[63:0]  = special (full 64-bit)
  import LSDecodeFields._
  val ls_default_decode = List(N, N, DADDR, DADDR, DSPECIAL, N)

  val ls_decode_list = ListLookup(
    func7,
    ls_default_decode,
    Array(
      MSET_BITPAT  -> List(
        N,
        N,
        0.U(memAddrLen.W),      // mem_addr: not used for MSET
        rs1(bankIdLen - 1, 0),  // bank_id from rs1[14:0]
        rs2,                    // special = full rs2
        Y
      ), // mset
      MVIN_BITPAT  -> List(
        Y,
        N,
        rs2(memAddrLen - 1, 0), // mem_addr from rs2[38:0]
        rs1(bankIdLen - 1, 0),  // bank_id from rs1[14:0]
        rs2,                    // special = full rs2
        Y
      ), // mvin
      MVOUT_BITPAT -> List(
        N,
        Y,
        rs2(memAddrLen - 1, 0), // mem_addr from rs2[38:0]
        rs1(bankIdLen - 1, 0),  // bank_id from rs1[14:0]
        rs2,                    // special = full rs2
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

  val raw_bank_id = rs1(bankIdLen - 1, 0)
  io.mem_decode_cmd_o.bits.is_shared := io.mem_decode_cmd_o.valid && (raw_bank_id > 31.U)
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
  // iter is always from rs1[63:48]
  io.mem_decode_cmd_o.bits.iter      := Mux(
    io.mem_decode_cmd_o.valid,
    rs1(63, 48),
    0.U(iterLen.W)
  )

  // Address parsing
  val ls_bank_id = ls_decode_list(LSDecodeFields.BANK_ID.id).asUInt
  io.mem_decode_cmd_o.bits.bank_id := Mux(io.mem_decode_cmd_o.valid, ls_bank_id, 0.U(log2Up(b.memDomain.bankNum).W))
  io.mem_decode_cmd_o.bits.special := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.SPECIAL.id).asUInt,
    0.U(64.W)
  )
}
