package framework.system.core

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import framework.system.core.rocket.RoCCIO
import framework.memdomain.isa.{MvoverISA, MvoverPort}
import framework.memdomain.frontend.mem.dma.{DmaError, DmaStatus}
import hier.core.rocket.{AdmissionPorts, CommandSnapshot}
import hier.tile.TaskAdmission
import memcore.bus.chi.{Params => ChiParams}
import memcore.memory.interlock.{Params => TrackingParams}

/** CPU-only controller command path: task RPC and bank moves have explicit completion owners. */
@instantiable
class ControllerAdmission(tracking: TrackingParams, bus: ChiParams, moves: Boolean)(implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  require(xLen == 64)

  @public
  val io = IO(new Bundle {
    val core        = Flipped(new AdmissionPorts(tracking, nPMPs, bus))
    val task        = Flipped(new RoCCIO(64))
    val taskSatp    = Output(UInt(64.W))
    val taskContext = Output(new CommandSnapshot(tracking, nPMPs))
    val move        = new MvoverPort
    val moveFault   = Valid(new DmaStatus)
  })

  val idle :: task :: moving :: moveResponse :: releasing :: Nil = Enum(5)
  val state                                                      = RegInit(idle)
  val tag                                                        = Reg(UInt(tracking.idBits.W))
  val rd                                                         = Reg(UInt(5.W))
  val failedMove                                                 = Reg(Bool())
  val instruction                                                = io.core.command.bits.instruction
  val isTask                                                     = instruction.opcode === "h2b".U
  val isMove                                                     = moves.B && instruction.opcode === "h7b".U && instruction.funct === MvoverISA.Funct.U
  val isFence                                                    = instruction.opcode === "h7b".U && instruction.funct === 0.U
  val controller: Instance[TaskAdmission] = Instantiate(new TaskAdmission(tracking, nPMPs))
  controller.io.task <> io.task
  io.taskSatp                 := controller.io.satp
  // The Ant launch handshake is the task command handshake; freeze this complete snapshot there.
  io.taskContext              := io.core.command.bits
  controller.io.workDrained   := true.B // This endpoint has no NPU or DMA client.
  controller.io.command.valid := state === idle && io.core.command.valid && isTask
  controller.io.command.bits  := io.core.command.bits

  io.core.reserve.ready           := !reset.asBool
  io.core.command.ready           := state === idle && !reset.asBool &&
    Mux(isTask, controller.io.command.ready, isFence || (isMove && io.move.command.ready))
  when(io.core.command.valid && state === idle && !reset.asBool) {
    assert(isTask || isMove || isFence, "CPU-only controller received an unsupported compute instruction")
  }
  io.move.command.valid           := state === idle && io.core.command.valid && isMove && !reset.asBool
  io.move.command.bits.sourceCore := instruction.rs1Data(7, 0)
  io.move.command.bits.targetCore := instruction.rs1Data(15, 8)
  io.move.command.bits.sourceBank := instruction.rs1Data(25, 16)
  io.move.command.bits.targetBank := instruction.rs1Data(35, 26)
  io.move.command.bits.sourceAddr := instruction.rs2Data(15, 0)
  io.move.command.bits.targetAddr := instruction.rs2Data(31, 16)
  io.move.command.bits.rows       := instruction.rs2Data(47, 32) +& 1.U
  when(io.core.command.fire) {
    tag   := io.core.command.bits.tag
    rd    := instruction.rd
    state := Mux(isTask, task, Mux(isMove, moving, releasing))
  }
  io.move.completion.ready        := state === moving && !reset.asBool
  when(io.move.completion.fire) {
    failedMove := io.move.completion.bits
    state      := moveResponse
  }
  io.moveFault.valid              := io.move.completion.fire && io.move.completion.bits
  io.moveFault.bits.error         := DmaError.Bank.U
  io.moveFault.bits.address       := 0.U // Existing move completion supplies no byte address.

  io.core.response.valid                                      := Mux(state === task, controller.io.response.valid, state === moveResponse) && !reset.asBool
  io.core.response.bits                                       := controller.io.response.bits
  when(state === moveResponse) {
    io.core.response.bits.rd   := rd
    io.core.response.bits.data := failedMove.asUInt
  }
  controller.io.response.ready                                := state === task && io.core.response.ready
  when(state === moveResponse && io.core.response.fire)(state := releasing)
  io.core.complete.valid                                      := (state === releasing || (state === task && controller.io.release.valid)) && !reset.asBool
  io.core.complete.bits.tag                                   := Mux(state === task, controller.io.release.bits.tag, tag)
  controller.io.release.ready                                 := state === task && io.core.complete.ready
  when(io.core.complete.fire)(state                           := idle)
  io.core.cancelled.valid                                     := false.B
  io.core.cancelled.bits                                      := 0.U.asTypeOf(io.core.cancelled.bits)
  io.core.interrupt                                           := false.B

  // A move/fence may have retired at CPU dispatch while its work is still pending here.
  // Do not let a following CPU store publish a completion flag before the bank move ends.
  val pendingMoveOrFence = io.core.command.valid && (isMove || isFence)
  io.core.cpuAllow          := !(pendingMoveOrFence || state === moving || state === moveResponse || state === releasing)
  io.core.cpuProbeAllow     := io.core.cpuAllow
  io.core.maintenance.valid := false.B
  io.core.maintenance.bits  := 0.U.asTypeOf(io.core.maintenance.bits)
  io.core.maintained.ready  := true.B
  io.core.pteRequest.valid  := false.B
  io.core.pteRequest.bits   := 0.U.asTypeOf(io.core.pteRequest.bits)
  io.core.pteResponse.ready := true.B
}
