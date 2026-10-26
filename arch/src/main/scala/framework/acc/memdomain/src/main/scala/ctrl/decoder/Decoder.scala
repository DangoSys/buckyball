package framework.acc.memdomain

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import framework.acc.frontend.decoder.{DomainId, PostGDCmd}

class DecodedCommand(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val is_shared      = Bool()
  val is_load        = Bool()
  val is_store       = Bool()
  val is_config      = Bool()
  val clear          = Bool()
  val transfer       = Bool()
  val source_bank_id = UInt(frontendParam.bank_id_len.W)
  val is_mvin_2d     = Bool()
  val is_image       = Bool()
  val is_mmio        = Bool()
  val mem_addr       = UInt(memParam.memAddrLen.W)
  val iter           = UInt(frontendParam.iter_len.W)
  val bank_id        = UInt(frontendParam.bank_id_len.W)
  val special        = UInt(64.W)
}

import framework.acc.memdomain.DISA._
import chisel3.experimental.hierarchy.{instantiable, public}

// LS decode fields (iter removed from decode table — always from rs1[63:48])
object LSDecodeFields extends Enumeration {
  type Field = Value
  val LD_EN, ST_EN, MEMADDR, BANK_ID, SPECIAL, VALID = Value
}

// Default constants for Mem decoder
object DecodeConstants {
  val Y        = true.B
  val N        = false.B
  val DADDR    = 0.U(15.W)
  val DSPECIAL = 0.U(64.W)
}

@instantiable
class Decoder(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Module {
  val commandDataBits = frontendParam.commandDataBits
  import DecodeConstants._

  val sharedBankIdBase = frontendParam.shared_bank_id_base
  require(sharedBankIdBase > 0, s"sharedBankIdBase($sharedBankIdBase) must be > 0")
  require(
    sharedBankIdBase < (1 << frontendParam.bank_id_len),
    s"sharedBankIdBase($sharedBankIdBase) must fit in bankIdLen(${frontendParam.bank_id_len})"
  )
  require(
    sharedBankIdBase > frontendParam.vbank_id_upper_bound,
    s"sharedBankIdBase($sharedBankIdBase) must be > vbankIdUpperBound(${frontendParam.vbank_id_upper_bound})"
  )

  @public
  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new PostGDCmd(frontendParam, memParam)))
    val command = Decoupled(new DecodedCommand(frontendParam, memParam))
  })

  val memAddrLen = memParam.memAddrLen
  val bankIdLen  = frontendParam.bank_id_len
  val iterLen    = frontendParam.iter_len

  // Only process Mem instructions
  io.request.ready := io.command.ready

  val func7 = io.request.bits.cmd.raw_inst(31, 25)
  val rs1   = io.request.bits.cmd.rs1Data
  val rs2   = io.request.bits.cmd.rs2Data

  // Unified encoding:
  //   rs1[9:0]   = read/config bank (BANK0)
  //   rs1[29:20] = write bank (BANK2)
  //   rs1[62:30] = iter; rs1[63] declares an in-place update.
  //   funct7[6:4] = enable (bank access flags, decoded by GlobalDecoder)
  //   rs2[38:0]  = mem_addr (for MVIN/MVOUT, 39-bit)
  //
  import LSDecodeFields._
  // format: off
  val ls_default_decode = List(N, N, DADDR, DADDR, DSPECIAL, N)
  val ls_decode_list = ListLookup(func7, ls_default_decode, Array(
      MSET_BITPAT      -> List(N, N, 0.U(memAddrLen.W),      rs1(bankIdLen - 1, 0), rs2, Y),
      MVIN_MMIO_BITPAT -> List(Y, N, rs2(memAddrLen - 1, 0), 0.U, rs2, Y),
      MVIN_BITPAT      -> List(Y, N, rs2(memAddrLen - 1, 0), rs1(bankIdLen + 19, 20), rs2, Y),
      MVIN_2D_BITPAT   -> List(Y, N, Cat(rs2(35, 0), 0.U(3.W)), rs1(bankIdLen + 19, 20), rs2, Y),
      MVIN_KERNEL_BITPAT -> List(Y, N, rs2(memAddrLen - 1, 0), rs1(bankIdLen + 19, 20), rs2, Y),
      MVOUT_BITPAT     -> List(N, Y, rs2(memAddrLen - 1, 0), rs1(bankIdLen - 1, 0), rs2, Y)
    )
  )
  // format: on

  assert(
    !(io.request.fire && !ls_decode_list(LSDecodeFields.VALID.id).asBool),
    s"Decoder: Invalid command opcode, func7 = 0x%x\n",
    func7
  )

// -----------------------------------------------------------------------------
// Output assignment
// -----------------------------------------------------------------------------
  io.command.valid := io.request.valid && (io.request.bits.domain_id === DomainId.MEM)

  val transfer    = func7 === MSET_BITPAT && rs2(12)
  val raw_bank_id = Mux(func7 === MVIN_BITPAT || func7 === MVIN_2D_BITPAT || transfer, rs1(29, 20), rs1(9, 0))

  val architectural_bank = Mux(
    func7 === MVIN_BITPAT || func7 === MVIN_2D_BITPAT || transfer,
    io.request.bits.bankAccess.wr_bank_id,
    io.request.bits.bankAccess.rd_bank_0_id
  )

  val config_bank    = Mux(func7 === MSET_BITPAT && !transfer, io.request.bits.bankAccess.wr_bank_id, architectural_bank)
  val shared_bank_id =
    memParam.sharedEnable.B && config_bank >= sharedBankIdBase.U
  // format: off
  // BB_BANK0 is encoded in rs1[9:0]. Keep the configured architectural ID;
  // private and shared banks occupy disjoint ranges in that namespace.
  io.command.bits.is_shared    := io.command.valid && shared_bank_id
  io.command.bits.is_load      := Mux(io.command.valid, ls_decode_list(LSDecodeFields.LD_EN.id).asBool, false.B)
  io.command.bits.is_store     := Mux(io.command.valid, ls_decode_list(LSDecodeFields.ST_EN.id).asBool, false.B)
  io.command.bits.is_config    := Mux(io.command.valid, func7 === MSET_BITPAT, false.B)
  io.command.bits.transfer     := io.command.valid && transfer
  io.command.bits.source_bank_id := rs1(bankIdLen - 1, 0)
  io.command.bits.clear        := io.command.valid && (func7 === MSET_BITPAT) && rs2(11)
  io.command.bits.is_mvin_2d   := Mux(io.command.valid, func7 === MVIN_2D_BITPAT, false.B)
  io.command.bits.is_mmio := func7 === MVIN_MMIO_BITPAT
  io.command.bits.is_image     := func7 === MVIN_KERNEL_BITPAT
  io.command.bits.mem_addr     := Mux(io.command.valid, ls_decode_list(LSDecodeFields.MEMADDR.id).asUInt, 0.U(memParam.memAddrLen.W))
  io.command.bits.iter         := Mux(io.command.valid, rs1(62, 30), 0.U(iterLen.W))

  when(io.request.fire && func7 === MSET_BITPAT) {
    when(transfer) {
      assert(rs1(63, 30) === 0.U && rs1(19, 10) === 0.U, "MSET transfer reserves rs1 outside BANK0/BANK2")
      assert(rs2 === (1.U << 12), "MSET transfer reserves other rs2 bits")
      assert(rs1(9, 0) =/= rs1(29, 20), "MSET transfer requires distinct source and target")
      assert((rs1(9, 0) >= sharedBankIdBase.U) === (rs1(29, 20) >= sharedBankIdBase.U), "MSET transfer requires one bank namespace")
    }.otherwise {
      assert(rs1(63, 10) === 0.U, "MSET reserves rs1[63:10]")
      assert(rs2(63, 12) === 0.U, "MSET reserves rs2[63:12]")
    }
    assert(!(rs2(11) && !rs2(10)), "MSET clear requires alloc=1")
  }
  when(io.request.fire && (func7 === MVIN_BITPAT || func7 === MVOUT_BITPAT)) {
    assert(rs2(63) || rs2(62, 58) === 0.U, "Whole-bank DMA reserves selected group bits")
  }
  when(io.request.fire && func7 === MVIN_2D_BITPAT) {
    val height     = rs1(62, 30)
    val pixelBytes = rs2(42, 36) << 3
    val width      = rs2(61, 59) +& 1.U
    val validBytes = Mux(rs2(62), 8.U(5.W), 16.U(5.W))
    assert(height =/= 0.U, "MVIN_2D height must be non-zero")
    assert(rs2(42, 36) =/= 0.U, "MVIN_2D pixel bytes must be non-zero")
    assert(rs2(52, 43) =/= 0.U, "MVIN_2D source width must be non-zero")
    assert(validBytes <= pixelBytes, "MVIN_2D valid bytes exceed pixel bytes")
    assert((rs2(58, 53) +& (height * width)) <= Mux(shared_bank_id, memParam.sharedBankEntries.U, memParam.bankEntries.U), "MVIN_2D destination exceeds bank")
    assert(!rs2(63), "MVIN_2D reserves rs2[63]")
  }
  when(io.request.fire && (func7 === MVIN_BITPAT || func7 === MVIN_2D_BITPAT)) {
    assert(rs1(19, 0) === 0.U, "MVIN write-bank encoding reserves rs1[19:0]")
  }

  val ls_bank_id = ls_decode_list(LSDecodeFields.BANK_ID.id).asUInt
  io.command.bits.bank_id := Mux(io.command.valid, Mux(transfer, raw_bank_id, ls_bank_id), 0.U(frontendParam.bank_id_len.W))
  io.command.bits.special := Mux(io.command.valid, ls_decode_list(LSDecodeFields.SPECIAL.id).asUInt, 0.U(64.W))
  // format: on
}
