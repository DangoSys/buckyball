package framework.memdomain.frontend.cmd_channel.decoder

import chisel3._
import chisel3.util._
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.frontend.decoder.PostGDCmd
import framework.memdomain.MemDomainParam
import framework.memdomain.frontend.cmd_channel.decoder.DISA._
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.Parameters
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}

// Detailed decode output for Mem domain
class MemDecodeCmd(parameter: MemDomainParam)(implicit p: Parameters) extends Bundle {
  val is_load  = Bool()
  val is_store = Bool()
  // Memory address
  val mem_addr = UInt(parameter.memAddrLen.W)
  // Iteration count
  val iter     = UInt(10.W)
  // Bank information
  // 3 bits, supports 8 banks (SPAD+ACC)
  val bank_id  = UInt(log2Up(parameter.bankNum).W)
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
class MemDomainDecoder(val parameter: MemDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {
  import MemDefaultConstants._

  @public
  val io = IO(new Bundle {
    val raw_cmd_i        = Flipped(Decoupled(new PostGDCmd))
    val mem_decode_cmd_o = Decoupled(new MemDecodeCmd(parameter))
  })

  val bankAddrLen = log2Up(parameter.bankEntries)
  val memAddrLen  = parameter.memAddrLen

  // Only process Mem instructions
  io.raw_cmd_i.ready := io.mem_decode_cmd_o.ready

  val func7 = io.raw_cmd_i.bits.raw_cmd.inst.funct
  val rs1   = io.raw_cmd_i.bits.raw_cmd.rs1
  val rs2   = io.raw_cmd_i.bits.raw_cmd.rs2

  // Load/Store instruction decoding
  import LSDecodeFields._
  val ls_default_decode = List(N, N, DADDR, DADDR, DITER, DSPECIAL, N)

  val ls_decode_list = ListLookup(
    func7,
    ls_default_decode,
    Array(
      MSET_BITPAT  -> List(
        Y,
        N,
        rs1(memAddrLen - 1, 0),
        rs2(bankAddrLen - 1, 0),
        rs2(bankAddrLen + 9, bankAddrLen),
        rs2(63, bankAddrLen + 10),
        Y
      ), // mset
      MVIN_BITPAT  -> List(
        Y,
        N,
        rs1(memAddrLen - 1, 0),
        rs2(bankAddrLen - 1, 0),
        rs2(bankAddrLen + 9, bankAddrLen),
        rs2(63, bankAddrLen + 10),
        Y
      ), // mvin
      MVOUT_BITPAT -> List(
        N,
        Y,
        rs1(memAddrLen - 1, 0),
        rs2(bankAddrLen - 1, 0),
        rs2(bankAddrLen + 9, bankAddrLen),
        rs2(63, bankAddrLen + 10),
        Y
      )  // mvout
    )
  )

  assert(
    !(io.raw_cmd_i.fire && !ls_decode_list(LSDecodeFields.VALID.id).asBool),
    s"MemDomainDecoder: Invalid command opcode, func7 = 0x%x\n",
    func7
  )

// -----------------------------------------------------------------------------
// Output assignment
// -----------------------------------------------------------------------------
  io.mem_decode_cmd_o.valid := io.raw_cmd_i.valid && (io.raw_cmd_i.bits.domain_id === DomainId.MEM)

  io.mem_decode_cmd_o.bits.is_load  := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.LD_EN.id).asBool,
    false.B
  )
  io.mem_decode_cmd_o.bits.is_store := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.ST_EN.id).asBool,
    false.B
  )
  io.mem_decode_cmd_o.bits.mem_addr := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.MEMADDR.id).asUInt,
    0.U(parameter.memAddrLen.W)
  )
  io.mem_decode_cmd_o.bits.iter     := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.ITER.id).asUInt,
    0.U(10.W)
  )

  // Address parsing
  val ls_bank_id = ls_decode_list(LSDecodeFields.BANK_ID.id).asUInt
  io.mem_decode_cmd_o.bits.bank_id := Mux(io.mem_decode_cmd_o.valid, ls_bank_id, 0.U(log2Up(parameter.bankNum).W))
  io.mem_decode_cmd_o.bits.special := Mux(
    io.mem_decode_cmd_o.valid,
    ls_decode_list(LSDecodeFields.SPECIAL.id).asUInt,
    0.U(40.W)
  )
}
