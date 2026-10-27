package framework.acc.frontend.decoder

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.frontend.scheduler.scoreboard.BankAccessInfo

import framework.acc.frontend.{Command, CommandPacket}
import framework.acc.memdomain.MemoryContext
import framework.acc.frontend.decoder.DISA._
import framework.acc.gpdomain.decoder.DISA._
import framework.acc.memdomain.DISA._

class PostGDCmd(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val commandDataBits = frontendParam.commandDataBits

  val runtime                  = Bool()
  val domain_id                = UInt(4.W)
  val iter                     = UInt(frontendParam.iter_len.W)
  val cmd                      = new Command(commandDataBits)
  val bankAccess               = new BankAccessInfo(frontendParam.bank_id_len)
  val architecturalDestination = UInt(frontendParam.bank_id_len.W)
  val op1_col                  = UInt(memParam.groupCountWidth.W)
  val op2_col                  = UInt(memParam.groupCountWidth.W)
  val wr_col                   = UInt(memParam.groupCountWidth.W)
  val wr_read_bank             = UInt(frontendParam.bank_id_len.W)
  val isBarrier                = Bool()
}

/** Only the frontend pipeline carries address-space context for later memory dispatch. */
class DecodedInstruction(frontendParam: FrontendParam, memParam: MemDomainParam, permissionParam: PmpParams)
    extends Bundle {
  val commandDataBits = frontendParam.commandDataBits
  val operation       = new PostGDCmd(frontendParam, memParam)
  val context         = new MemoryContext(permissionParam)
}

@instantiable
class GlobalDecoder(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val permissionParam: PmpParams)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits

  val bankIdLen = frontendParam.bank_id_len

  @public
  val io = IO(new Bundle {

    val command = Flipped(Decoupled(new CommandPacket(permissionParam, frontendParam)))

    val decoded = Decoupled(new DecodedInstruction(frontendParam, memParam, permissionParam))
  })

  // Decoding follows the consumer handshake.
  io.command.ready := io.decoded.ready

  val func7  = io.command.bits.instruction.raw_inst(31, 25)
  val opcode = io.command.bits.instruction.raw_inst(6, 0)
  val rs1    = io.command.bits.instruction.rs1Data

  // Route memory and GP commands to their owning domains.
  val is_mem_inst = (func7 === MVIN_MMIO_BITPAT) || (func7 === MVIN_BITPAT) ||
    (func7 === MVIN_2D_BITPAT) ||
    (func7 === MVOUT_BITPAT) ||
    (func7 === MSET_BITPAT) ||
    (func7 === MVOVER_BITPAT) ||
    (func7 === MVIN_KERNEL_BITPAT)

  val is_barrier_inst = func7 === BARRIER_BITPAT

  val is_kernel_inst = func7 === RUN_KERNEL_BITPAT
  val is_bare_rvv    = opcode === "h57".U || opcode === "h07".U || opcode === "h27".U
  val is_ball_inst   = !is_mem_inst && !is_barrier_inst && !is_bare_rvv && !is_kernel_inst

  when(io.command.fire) {
    assert(func7 =/= 0.U, "GlobalDecoder: funct7 zero is not a Buckyball instruction")
    assert(!is_bare_rvv, "GlobalDecoder: bare RVV instructions are not Buckyball commands")
    when(is_kernel_inst) {
      assert(gpParam.enable.B, "GlobalDecoder: kernel command requires gpDomain.enable=true")
    }
  }

  // Encode domain ID
  val domain_id = MuxCase(
    DomainId.BALL,
    Seq(
      is_barrier_inst -> DomainId.FRONTEND,
      is_kernel_inst  -> DomainId.GP,
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
  //   rs1[62:30] = iter; rs1[63] declares an in-place update.
  //
  // funct7[6:4] enable encoding:
  //   000 = no bank access
  //   001 = 1 read (bank0)
  //   010 = 1 write (bank2)
  //   011 = 1 read + 1 write (bank0 read, bank2 write)
  //   100 = 2 read + 1 write (bank0+bank1 read, bank2 write)
  //   101,110,111 = no bank access (extended opcode space)
  // -------------------------------------------------------------------------
  val bankAccess = Wire(new BankAccessInfo(frontendParam.bank_id_len))
  val enableBits = func7(6, 4)

  // Decode enable from funct7[6:4]
  val msetTransfer = func7 === MSET_BITPAT && io.command.bits.instruction.rs2Data(12)
  val hasRd0       = enableBits === 1.U || enableBits === 3.U || enableBits === 4.U || msetTransfer
  val hasRd1       = enableBits === 4.U || msetTransfer
  val hasWr        = (enableBits === 2.U || enableBits === 3.U || enableBits === 4.U) && func7 =/= MVIN_MMIO_BITPAT

  bankAccess.rd_bank_0_valid := hasRd0
  bankAccess.rd_bank_0_id    := rs1(bankIdLen - 1, 0)
  bankAccess.rd_bank_1_valid := hasRd1
  bankAccess.rd_bank_1_id    := Mux(msetTransfer, rs1(bankIdLen + 19, 20), rs1(bankIdLen + 9, 10))
  bankAccess.wr_bank_valid   := hasWr

  val inplaceUpdate = hasWr && func7 =/= MSET_BITPAT &&
    (rs1(
      63
    ) || is_kernel_inst || func7 === MVIN_2D_BITPAT || (func7 === MVIN_BITPAT && io.command.bits.instruction.rs2Data(63)))

  val readOldWrite = hasWr && func7 =/= MSET_BITPAT && (rs1(29) || inplaceUpdate)
  val writeBank    = rs1(28, 20).pad(bankIdLen)
  bankAccess.wr_bank_id  := Mux(
    func7 === MSET_BITPAT,
    Mux(msetTransfer, rs1(bankIdLen + 19, 20), rs1(bankIdLen - 1, 0)),
    writeBank
  )
  bankAccess.rd_wr_valid := readOldWrite
  bankAccess.rd_wr_id    := writeBank
  bankAccess.wr_update   := inplaceUpdate

  def legalBank(raw: UInt): Bool =
    raw <= frontendParam.max_private_vbank.U ||
      (gpParam.enable.B && raw >= memParam.virtualBankCount.U && raw < (memParam.virtualBankCount + 2).U) ||
      (memParam.sharedEnable.B && raw >= frontendParam.shared_vbank_base.U &&
        raw < memParam.virtualBankCount.U)

  val usesArchitecturalBank = is_ball_inst || func7 === MSET_BITPAT || func7 === MVIN_BITPAT ||
    func7 === MVIN_2D_BITPAT || func7 === MVOUT_BITPAT
  when(io.command.fire && usesArchitecturalBank) {
    when(hasRd0)(assert(legalBank(bankAccess.rd_bank_0_id), "GlobalDecoder: bank0 is outside configured ranges"))
    when(hasRd1)(assert(legalBank(bankAccess.rd_bank_1_id), "GlobalDecoder: bank1 is outside configured ranges"))
    when(hasWr)(assert(legalBank(bankAccess.wr_bank_id), "GlobalDecoder: write bank is outside configured ranges"))
  }

  // Output control
  io.decoded.bits.context                            := io.command.bits.context
  io.decoded.bits.operation.runtime                  := io.command.bits.runtime
  io.decoded.valid                                   := io.command.valid
  io.decoded.bits.operation.domain_id                := domain_id
  io.decoded.bits.operation.iter                     := rs1(62, 30)
  io.decoded.bits.operation.cmd                      := io.command.bits.instruction
  io.decoded.bits.operation.bankAccess               := bankAccess
  io.decoded.bits.operation.architecturalDestination := bankAccess.wr_bank_id
  io.decoded.bits.operation.op1_col                  := 0.U
  io.decoded.bits.operation.op2_col                  := 0.U
  io.decoded.bits.operation.wr_col                   := 0.U
  io.decoded.bits.operation.wr_read_bank             := 0.U
  io.decoded.bits.operation.isBarrier                := is_barrier_inst
}
