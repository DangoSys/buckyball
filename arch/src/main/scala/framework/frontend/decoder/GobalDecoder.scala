package framework.frontend.decoder

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.frontend.scoreboard.BankAccessInfo
import freechips.rocketchip.tile._

import framework.frontend.decoder.GISA._
import framework.memdomain.frontend.cmd.decoder.DISA._

import framework.system.core.rocket.RoCCCommandBB

@instantiable
class GlobalDecoder(val b: GlobalConfig) extends Module {

  val bankIdLen = b.frontend.bank_id_len

  @public
  val io = IO(new Bundle {

    val id_i = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommandBB(b.tile.xLen)
    }))

    val id_o = Decoupled(new PostGDCmd(b))
  })

  // If reservation station is blocked, id_i is also blocked
  io.id_i.ready := io.id_o.ready

  val func7  = io.id_i.bits.cmd.funct
  val funct3 = io.id_i.bits.cmd.funct3
  val opcode = io.id_i.bits.cmd.opcode
  val rs1    = io.id_i.bits.cmd.rs1Data

  // Private kernel commands are issued by the RVV controller.
  val is_mem_inst = (func7 === MVIN_BITPAT) ||
    (func7 === MVIN_2D_BITPAT) ||
    (func7 === MVOUT_BITPAT) ||
    (func7 === MSET_BITPAT) ||
    (func7 === MVIN_MMIO_BITPAT) ||
    (func7 === MVOVER_BITPAT)

  val is_barrier_inst = func7 === BARRIER_BITPAT

  val programBank      = rs1(bankIdLen - 1, 0) >= b.memDomain.virtualBankCount.U &&
    rs1(bankIdLen - 1, 0) < (b.memDomain.virtualBankCount + 2).U
  val isProgramRelease = b.rvv.enable.B && func7 === MSET_BITPAT && programBank
  val is_kernel_inst   = func7 === MVIN_KERNEL_BITPAT || func7 === RUN_KERNEL_BITPAT || isProgramRelease
  val is_bare_rvv      = opcode === "h57".U || opcode === "h07".U || opcode === "h27".U
  val is_ball_inst     = !is_mem_inst && !is_barrier_inst && !is_bare_rvv && !is_kernel_inst

  when(io.id_i.fire) {
    assert(func7 =/= 0.U, "GlobalDecoder: funct7 zero is not a Buckyball instruction")
    assert(!is_bare_rvv, "GlobalDecoder: bare RVV instructions are not Buckyball commands")
    when(is_kernel_inst) {
      assert(b.rvv.enable.B, "GlobalDecoder: kernel command requires rvv.enable=true")
    }
  }

  // Encode domain ID
  val domain_id = MuxCase(
    DomainId.BALL,
    Seq(
      is_barrier_inst -> DomainId.FRONTEND,
      is_kernel_inst  -> DomainId.RVV,
      is_mem_inst     -> DomainId.MEM,
      is_ball_inst    -> DomainId.BALL
    )
  )

  // -------------------------------------------------------------------------
  // Bank access info extraction — enable flags from funct7[6:4]
  //
  // Unified rs1 layout (defined in isa.h):
  //   rs1[9:0]   = bank_0  (rd_bank_0 or MSET config bank)
  //   rs1[19:10] = bank_1  (rd_bank_1, dual-operand only)
  //   rs1[29:20] = bank_2  (wr_bank for Ball instructions)
  //   rs1[63:30] = iter (34-bit)
  //
  // funct7[6:4] enable encoding:
  //   000 = no bank access
  //   001 = 1 read (bank0)
  //   010 = 1 write (bank2)
  //   011 = 1 read + 1 write (bank0 read, bank2 write)
  //   100 = 2 read + 1 write (bank0+bank1 read, bank2 write)
  //   101,110,111 = no bank access (extended opcode space)
  // -------------------------------------------------------------------------
  val bankAccess = Wire(new BankAccessInfo(bankIdLen))
  val enableBits = func7(6, 4)

  // Decode enable from funct7[6:4]
  val msetTransfer = func7 === MSET_BITPAT && io.id_i.bits.cmd.rs2Data(12)
  val hasRd0       = enableBits === 1.U || enableBits === 3.U || enableBits === 4.U || msetTransfer
  val hasRd1       = enableBits === 4.U
  val hasWr        = (enableBits === 2.U || enableBits === 3.U || enableBits === 4.U) && func7 =/= MVIN_MMIO_BITPAT

  val ballBid = WireDefault(0.U(5.W))
  b.ballDomain.ballISA.foreach { entry =>
    when(func7 === entry.funct7.U) {
      ballBid := entry.bid.U
    }
  }

  bankAccess.rd_bank_0_valid := hasRd0
  bankAccess.rd_bank_0_id    := rs1(bankIdLen - 1, 0)
  bankAccess.rd_bank_1_valid := hasRd1
  bankAccess.rd_bank_1_id    := rs1(bankIdLen + 9, 10)
  bankAccess.wr_bank_valid   := hasWr
  bankAccess.wr_bank_id      := Mux(func7 === MSET_BITPAT && !msetTransfer, rs1(bankIdLen - 1, 0), rs1(bankIdLen + 19, 20))

  private def legalBank(raw: UInt): Bool =
    raw <= b.frontend.vbank_id_upper_bound.U ||
      (b.rvv.enable.B && raw >= b.memDomain.virtualBankCount.U && raw < (b.memDomain.virtualBankCount + 2).U) ||
      (b.memDomain.sharedEnable.B && raw >= b.frontend.shared_bank_id_base.U &&
        raw < b.memDomain.virtualBankCount.U)

  val usesArchitecturalBank = is_ball_inst || func7 === MSET_BITPAT || func7 === MVIN_BITPAT ||
    func7 === MVIN_2D_BITPAT || func7 === MVOUT_BITPAT
  when(io.id_i.fire && usesArchitecturalBank) {
    when(hasRd0)(assert(legalBank(bankAccess.rd_bank_0_id), "GlobalDecoder: bank0 is outside configured ranges"))
    when(hasRd1)(assert(legalBank(bankAccess.rd_bank_1_id), "GlobalDecoder: bank1 is outside configured ranges"))
    when(hasWr)(assert(legalBank(bankAccess.wr_bank_id), "GlobalDecoder: write bank is outside configured ranges"))
  }

  // Output control
  io.id_o.valid           := io.id_i.valid
  io.id_o.bits.domain_id  := domain_id
  io.id_o.bits.ball_bid   := ballBid
  io.id_o.bits.cmd        := io.id_i.bits.cmd
  io.id_o.bits.bankAccess := bankAccess
  io.id_o.bits.op1_col    := 0.U
  io.id_o.bits.op2_col    := 0.U
  io.id_o.bits.wr_col     := 0.U
  io.id_o.bits.isBarrier  := is_barrier_inst
}
