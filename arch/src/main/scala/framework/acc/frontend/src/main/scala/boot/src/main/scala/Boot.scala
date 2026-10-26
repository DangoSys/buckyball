package framework.acc.frontend.boot

import framework.acc.frontend.configs.FrontendParam
import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.acc.balldomain.decoder.DISA.BALL_INIT_BITPAT
import framework.acc.frontend.CommandPacket

@instantiable
class BootRom(val ballIds: Seq[Int], val permissionParam: PmpParams, val fp: FrontendParam) extends Module {
  val commandDataBits = fp.commandDataBits

  val commandBits = commandDataBits

  @public
  val io = IO(new Bundle {

    val command       = Flipped(Decoupled(new CommandPacket(permissionParam, fp)))
    val cmd           = Decoupled(new CommandPacket(permissionParam, fp))
    val schedulerIdle = Input(Bool())
    val idle          = Output(Bool())
  })

  val ballInitRecords = ballIds.map { id =>
    require(id >= 0 && id < 32, s"BALL_INIT target ID out of range: $id")
    BuckyballCommand(BALL_INIT_BITPAT.value.toInt, rs1 = 0, rs2 = BigInt(id))
  }

  val bootRecords = ballInitRecords

  val bootFuncts  = VecInit(bootRecords.map(r => r.funct.U(7.W)))
  val bootRs1Data = VecInit(bootRecords.map(r => r.rs1.U(commandBits.W)))
  val bootRs2Data = VecInit(bootRecords.map(r => r.rs2.U(commandBits.W)))
  val bootPcWidth = math.max(1, log2Ceil(bootRecords.length + 1))

  val active   = RegInit(true.B)
  val drain    = RegInit(false.B)
  val waitIdle = RegInit(false.B)
  val pc       = RegInit(0.U(bootPcWidth.W))

  val atEnd = pc === bootRecords.length.U(bootPcWidth.W)

  val current = Wire(new CommandPacket(permissionParam, fp))
  current                      := 0.U.asTypeOf(new CommandPacket(permissionParam, fp))
  current.instruction.raw_inst := Cat(
    Mux(atEnd, 0.U(7.W), bootFuncts(pc)),
    0.U(18.W),
    BuckyballCommand.Custom3Opcode.U(7.W)
  )
  current.instruction.rs1Data  := Mux(atEnd, 0.U, bootRs1Data(pc))
  current.instruction.rs2Data  := Mux(atEnd, 0.U, bootRs2Data(pc))

  val injectValid = active && !drain && !waitIdle && !atEnd

  when(active && !drain && atEnd) {
    drain := true.B
  }
  when(waitIdle && io.schedulerIdle) {
    waitIdle := false.B
  }
  when(drain && io.schedulerIdle) {
    active := false.B
  }
  when(active && io.cmd.fire) {
    pc       := pc + 1.U
    waitIdle := true.B
  }

  // External commands enter only after all reset-time initialization has retired.
  io.cmd.valid     := Mux(active, injectValid, io.command.valid)
  io.cmd.bits      := Mux(active, current, io.command.bits)
  io.command.ready := !active && io.cmd.ready
  io.idle          := !active && io.schedulerIdle && !io.command.valid

  when(io.command.fire) {
    assert(!io.command.bits.instruction.raw_inst(14), "Frontend commands must not request a register response")
  }
}

/**
 * Elaboration-time representation of one Buckyball custom instruction.
 * Domain ISA builders own the funct7 and operand encodings; boot only streams
 * these already-formed commands.
 */
case class BuckyballCommand(funct: Int, rs1: BigInt = 0, rs2: BigInt = 0)

object BuckyballCommand {
  val Custom3Opcode = 0x7b
  val Custom3Funct3 = 3
}
