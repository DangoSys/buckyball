package framework.frontend

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.system.core.rocket.RoCCCommandBB
import framework.top.GlobalConfig

object BootAddress {
  val ZeroBase: BigInt = (BigInt(1) << 39) - (BigInt(1) << 20)

  def isZeroBase(addr: UInt): Bool = addr === ZeroBase.U(addr.getWidth.W)
}

@instantiable
class BootRom(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {

    val cmd = Decoupled(new Bundle {
      val cmd = new RoCCCommandBB(b.core.xLen)
    })

    val schedulerIdle = Input(Bool())
    val active        = Output(Bool())
  })

  private val custom3Opcode = 0x7b
  private val custom3Funct3 = 3
  private val msetFunct     = 0x20
  private val mvinFunct     = 0x21
  private val initEndFunct  = 2

  private val bankFieldWidth = 10
  private val iterShift      = 30
  private val strideShift    = 39
  private val msetColShift   = 5
  private val msetAllocBit   = 10

  private case class BootRecord(funct: Int, rs1: BigInt, rs2: BigInt)

  private def bootCmd(funct: Int, rs1: BigInt, rs2: BigInt): BootRecord =
    BootRecord(funct, rs1, rs2)

  private def bankField(bankId: Int): BigInt = BigInt(bankId) & ((BigInt(1) << bankFieldWidth) - 1)

  private def msetAlloc(bankId: Int, cols: Int): BootRecord = {
    val rs2 = (BigInt(cols) << msetColShift) | (BigInt(1) << msetAllocBit)
    bootCmd(msetFunct, bankField(bankId), rs2)
  }

  private def msetRelease(bankId: Int): BootRecord =
    bootCmd(msetFunct, bankField(bankId), 0)

  private def mvinZero(bankId: Int): BootRecord = {
    val rs1 = bankField(bankId) | (BigInt(b.memDomain.bankEntries) << iterShift)
    val rs2 = BootAddress.ZeroBase | (BigInt(1) << strideShift)
    bootCmd(mvinFunct, rs1, rs2)
  }

  require(b.memDomain.bankNum <= 32, "mset column field is 5 bits in the current ISA encoding")

  private val bootBankId = 0

  private val bootRecords =
    Seq(
      msetAlloc(bootBankId, b.memDomain.bankNum),
      mvinZero(bootBankId),
      msetRelease(bootBankId),
      bootCmd(initEndFunct, 0, 0)
    )

  private val bootFuncts  = VecInit(bootRecords.map(r => r.funct.U(7.W)))
  private val bootRs1Data = VecInit(bootRecords.map(r => r.rs1.U(b.core.xLen.W)))
  private val bootRs2Data = VecInit(bootRecords.map(r => r.rs2.U(b.core.xLen.W)))
  private val bootPcWidth = math.max(1, log2Ceil(bootRecords.length))

  private val active   = RegInit(true.B)
  private val drain    = RegInit(false.B)
  private val waitIdle = RegInit(false.B)
  private val pc       = RegInit(0.U(bootPcWidth.W))

  private val current = Wire(new RoCCCommandBB(b.core.xLen))
  current         := 0.U.asTypeOf(new RoCCCommandBB(b.core.xLen))
  current.funct   := bootFuncts(pc)
  current.funct3  := custom3Funct3.U
  current.opcode  := custom3Opcode.U
  current.rs1Data := bootRs1Data(pc)
  current.rs2Data := bootRs2Data(pc)

  private val atEnd       = bootFuncts(pc) === initEndFunct.U
  private val injectValid = active && !drain && !waitIdle && !atEnd

  when(active && !drain && atEnd) {
    drain := true.B
  }
  when(waitIdle && io.schedulerIdle) {
    waitIdle := false.B
  }
  when(drain && io.schedulerIdle) {
    active := false.B
  }
  when(io.cmd.fire) {
    pc       := pc + 1.U
    waitIdle := true.B
  }

  io.cmd.valid    := injectValid
  io.cmd.bits.cmd := current
  io.active       := active
}
