package framework.memdomain.frontend.cmd.decoder

import chisel3._
import chisel3.util._
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd.decoder.DISA._
import freechips.rocketchip.tile._
import chisel3.experimental.hierarchy.{instantiable, public}

// Detailed decode output for Mem domain
class MemDecodeCmd(b: GlobalConfig) extends Bundle {
  // Shared memory access marker: raw BB_BANK0 id >= frontend.sharedBankIdBase.
  val is_shared = Bool()
  val is_load   = Bool()
  val is_store  = Bool()
  val is_config = Bool()
  val clear     = Bool()

  val is_mvin_mmio = Bool()

  val mem_addr = UInt(b.memDomain.memAddrLen.W)
  val iter     = UInt(b.frontend.iter_len.W)
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

  private val sharedBankIdBase = b.frontend.shared_bank_id_base
  private val bankIdxAlign     = 1 << log2Ceil(b.memDomain.bankNum)
  require(sharedBankIdBase > 0, s"sharedBankIdBase($sharedBankIdBase) must be > 0")
  require(
    sharedBankIdBase < (1 << b.frontend.bank_id_len),
    s"sharedBankIdBase($sharedBankIdBase) must fit in bankIdLen(${b.frontend.bank_id_len})"
  )
  require(
    sharedBankIdBase > b.frontend.vbank_id_upper_bound,
    s"sharedBankIdBase($sharedBankIdBase) must be > vbankIdUpperBound(${b.frontend.vbank_id_upper_bound})"
  )
  require(
    sharedBankIdBase >= b.memDomain.bankNum,
    s"sharedBankIdBase($sharedBankIdBase) must be >= bankNum(${b.memDomain.bankNum})"
  )
  require(
    sharedBankIdBase % bankIdxAlign == 0,
    s"sharedBankIdBase($sharedBankIdBase) must be aligned to bank index width 2^ceil(log2(bankNum))=$bankIdxAlign"
  )

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
  //   rs1[9:0]   = bank_id (BANK0)
  //   rs1[63:30] = iter (34-bit)
  //   funct7[6:4] = enable (bank access flags, decoded by GlobalDecoder)
  //   rs2[38:0]  = mem_addr (for MVIN/MVOUT, 39-bit)
  //
  // mvin_mmio: rs1[63:30]=row (iter), rs2[38:0]=dram_addr, rs2[55:39]=mmio_addr, rs2[63:56]=col
  import LSDecodeFields._
  // format: off
  val ls_default_decode = List(N, N, DADDR, DADDR, DSPECIAL, N)
  val ls_decode_list = ListLookup(func7, ls_default_decode, Array(
      MSET_BITPAT      -> List(N, N, 0.U(memAddrLen.W),      rs1(bankIdLen - 1, 0), rs2, Y),
      MVIN_BITPAT      -> List(Y, N, rs2(memAddrLen - 1, 0), rs1(bankIdLen - 1, 0), rs2, Y),
      MVOUT_BITPAT     -> List(N, Y, rs2(memAddrLen - 1, 0), rs1(bankIdLen - 1, 0), rs2, Y),
      MVIN_MMIO_BITPAT -> List(Y, N, rs2(memAddrLen - 1, 0), 0.U(bankIdLen.W),      rs2, Y)
    )
  )
  // format: on

  assert(
    !(io.cmd_i.fire && !ls_decode_list(LSDecodeFields.VALID.id).asBool),
    s"MemDomainDecoder: Invalid command opcode, func7 = 0x%x\n",
    func7
  )

// -----------------------------------------------------------------------------
// Output assignment
// -----------------------------------------------------------------------------
  io.mem_decode_cmd_o.valid := io.cmd_i.valid && (io.cmd_i.bits.domain_id === DomainId.MEM)

  val raw_bank_id = rs1(9, 0)
  // format: off
  // BB_BANK0 is encoded in rs1[9:0]. Use the full raw field to detect
  // shared banks (id >= sharedBankIdBase) before truncating to local vbank width.
  io.mem_decode_cmd_o.bits.is_shared    := io.mem_decode_cmd_o.valid && (raw_bank_id >= sharedBankIdBase.U)
  io.mem_decode_cmd_o.bits.is_load      := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.LD_EN.id).asBool, false.B)
  io.mem_decode_cmd_o.bits.is_store     := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.ST_EN.id).asBool, false.B)
  io.mem_decode_cmd_o.bits.is_config    := Mux(io.mem_decode_cmd_o.valid, func7 === MSET_BITPAT, false.B)
  io.mem_decode_cmd_o.bits.clear        := io.mem_decode_cmd_o.valid && (func7 === MSET_BITPAT) && rs2(11)
  io.mem_decode_cmd_o.bits.is_mvin_mmio := Mux(io.mem_decode_cmd_o.valid, func7 === MVIN_MMIO_BITPAT, false.B)
  io.mem_decode_cmd_o.bits.mem_addr     := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.MEMADDR.id).asUInt, 0.U(b.memDomain.memAddrLen.W))
  io.mem_decode_cmd_o.bits.iter         := Mux(io.mem_decode_cmd_o.valid, rs1(63, 30), 0.U(iterLen.W))

  when(io.cmd_i.fire && func7 === MSET_BITPAT) {
    assert(rs1(63, 10) === 0.U, "MSET reserves rs1[63:10]")
    assert(rs2(63, 12) === 0.U, "MSET reserves rs2[63:12]")
    assert(!(rs2(11) && !rs2(10)), "MSET clear requires alloc=1")
  }

  val ls_bank_id = ls_decode_list(LSDecodeFields.BANK_ID.id).asUInt
  io.mem_decode_cmd_o.bits.bank_id := Mux(io.mem_decode_cmd_o.valid, ls_bank_id, 0.U(log2Up(b.memDomain.bankNum).W))
  io.mem_decode_cmd_o.bits.special := Mux(io.mem_decode_cmd_o.valid, ls_decode_list(LSDecodeFields.SPECIAL.id).asUInt, 0.U(64.W))
  // format: on
}
