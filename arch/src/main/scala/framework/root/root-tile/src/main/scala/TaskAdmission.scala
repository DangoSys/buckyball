package hier.tile

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import hier.core.rocket.CommandSnapshot
import framework.system.core.rocket.{RoCCIO, RoCCResponseBB}
import memcore.memory.interlock.{Params => TrackingParams, Tag}

/** Routes an accepted task command without allocating an accelerator ROB entry. */
@instantiable
class TaskAdmission(tracking: TrackingParams, pmps: Int)(implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {

  @public val io = IO(new Bundle {
    val command     = Flipped(Decoupled(new CommandSnapshot(tracking, pmps)))
    val task        = Flipped(new RoCCIO(64))
    val response    = Decoupled(new RoCCResponseBB(64))
    val release     = Decoupled(new Tag(tracking))
    val satp        = Output(UInt(64.W))
    // Worker completion cannot publish a result while its accepted NPU/DMA/cache work remains live.
    val workDrained = Input(Bool())
  })

  val idle :: responding :: releasing :: Nil = Enum(3)
  val state                                  = RegInit(idle)
  val tag                                    = Reg(UInt(tracking.idBits.W))
  val satp                                   = Reg(UInt(64.W))
  val finishTask                             = io.command.bits.instruction.funct === 2.U
  val eligible                               = state === idle && (!finishTask || io.workDrained) && !reset.asBool
  io.task.cmd.valid            := io.command.valid && eligible
  io.task.cmd.bits             := io.command.bits.instruction
  io.command.ready             := io.task.cmd.ready && eligible
  io.task.exception            := false.B
  when(io.command.fire) {
    assert(io.command.bits.instruction.opcode === "h2b".U, "TaskAdmission received a compute instruction")
    tag   := io.command.bits.tag
    satp  := io.command.bits.satp
    state := responding
  }
  // TaskController samples the root on the command handshake, including the first staged field.
  io.satp                      := Mux(state === idle && io.command.valid, io.command.bits.satp, satp)
  io.response.valid            := state === responding && io.task.resp.valid && !reset.asBool
  io.response.bits             := io.task.resp.bits
  io.task.resp.ready           := state === responding && io.response.ready && !reset.asBool
  when(io.response.fire)(state := releasing)
  io.release.valid             := state === releasing && !reset.asBool
  io.release.bits.tag          := tag
  when(io.release.fire)(state  := idle)
}
