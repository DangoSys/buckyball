package framework.tlp.controller.admission

import framework.generic.contract.controller.{CommandSnapshot, Commands, Management, TaskEndpoint}
import framework.generic.contract.memdomain.MvoverISA
import framework.generic.contract.memdomain.MvoverPort

import framework.generic.memory.interlock.{Tag, Params => TrackingParam}
import framework.tlp.controller.configs.CommandsParam
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.contract.frontend.Response
import framework.generic.contract.frontend.RoCCIO
import framework.generic.contract.frontend.Command
import framework.generic.contract.memory.access.preflight.PmpEntry
import framework.generic.contract.memory.access.preflight.configs.PmpParams

/** CPU-only controller command path: task RPC and bank moves have explicit completion owners. */
@instantiable
class CommandAdmission(b: CommandsParam) extends Module {
  val tracking = b.tracking
  val moves    = b.moves

  @public
  val io = IO(new Bundle {
    val execution =
      Flipped(new Commands(tracking, b.dataBits, b.permission))

    val taskRequest = Decoupled(new CommandSnapshot(
      tracking,
      b.dataBits,
      b.permission
    ))

    val taskResponse  = Flipped(Decoupled(new Response(b.dataBits)))
    val move          = new MvoverPort
    val memoryAllowed = Output(Bool())
    val idle          = Output(Bool())
  })

  val idle :: task :: moving :: moveResponse :: releasing :: Nil = Enum(5)
  val state                                                      = RegInit(idle)
  val tag                                                        = Reg(UInt(tracking.idBits.W))
  val rd                                                         = Reg(UInt(5.W))
  val failedMove                                                 = Reg(Bool())
  val instruction                                                = io.execution.command.bits.instruction
  val isTask                                                     = instruction.raw_inst(6, 0) === "h2b".U
  val isMove                                                     = moves.B && instruction.raw_inst(6, 0) === "h7b".U && instruction.raw_inst(31, 25) === MvoverISA.Funct.U

  val writeResult = Reg(Bool())
  io.taskRequest.valid := state === idle && io.execution.command.valid && isTask && !reset.asBool
  io.taskRequest.bits  := io.execution.command.bits

  io.execution.command.ready        := state === idle && !reset.asBool &&
    Mux(isTask, io.taskRequest.ready, isMove && io.move.command.ready)
  when(io.execution.command.valid && state === idle && !reset.asBool) {
    assert(
      b.taskEnabled.B && (isTask || isMove),
      "CPU-only controller received an unsupported compute instruction"
    )
  }
  io.move.command.valid             := state === idle && io.execution.command.valid && isMove && !reset.asBool
  io.move.command.bits.sourceThread := instruction.rs1Data(7, 0)
  io.move.command.bits.targetThread := instruction.rs1Data(15, 8)
  io.move.command.bits.sourceBank   := instruction.rs1Data(25, 16)
  io.move.command.bits.targetBank   := instruction.rs1Data(35, 26)
  io.move.command.bits.sourceAddr   := instruction.rs2Data(15, 0)
  io.move.command.bits.targetAddr   := instruction.rs2Data(31, 16)
  io.move.command.bits.rows         := instruction.rs2Data(47, 32) +& 1.U

  when(io.execution.command.fire) {
    writeResult := instruction.raw_inst(14)
    tag         := io.execution.command.bits.tag
    rd          := instruction.raw_inst(11, 7)
    state       := Mux(isTask, task, moving)
  }
  io.move.completion.ready := state === moving && !reset.asBool
  when(io.move.completion.fire) {
    failedMove := io.move.completion.bits
    state      := Mux(writeResult, moveResponse, releasing)
  }

  io.execution.response.valid                                                                := ((state === task && io.taskResponse.valid && writeResult) || state === moveResponse) && !reset.asBool
  io.execution.response.bits                                                                 := io.taskResponse.bits
  when(state === moveResponse) {
    io.execution.response.bits.rd   := rd
    io.execution.response.bits.data := failedMove.asUInt
  }
  io.taskResponse.ready                                                                      := state === task && (!writeResult || io.execution.response.ready) && !reset.asBool
  when(io.taskResponse.fire || (state === moveResponse && io.execution.response.fire))(state := releasing)
  io.execution.complete.valid                                                                := state === releasing && !reset.asBool
  io.execution.complete.bits.tag                                                             := tag
  when(io.execution.complete.fire)(state                                                     := idle)
  // A move may have retired at CPU dispatch while its work is still pending here.
  // Do not let a following CPU store publish a completion flag before the bank move ends.
  val pendingMove = io.execution.command.valid && isMove
  io.memoryAllowed := !(pendingMove || state === moving || state === moveResponse || state === releasing)
  io.idle          := state === idle
}

/** One posted move command remains busy until the shared fabric reports completion. */
@instantiable
class MoveAdmission(dataBits: Int, permission: PmpParams) extends Module {

  @public
  val io = IO(new Bundle {
    val rocc = new RoCCIO(dataBits, permission)
    val move = new MvoverPort
  })

  val moving      = RegInit(false.B)
  val command     = io.rocc.cmd
  val instruction = command.bits.instruction
  io.move.command.valid                := command.valid && !moving
  io.move.command.bits.sourceThread    := instruction.rs1Data(7, 0)
  io.move.command.bits.targetThread    := instruction.rs1Data(15, 8)
  io.move.command.bits.sourceBank      := instruction.rs1Data(25, 16)
  io.move.command.bits.targetBank      := instruction.rs1Data(35, 26)
  io.move.command.bits.sourceAddr      := instruction.rs2Data(15, 0)
  io.move.command.bits.targetAddr      := instruction.rs2Data(31, 16)
  io.move.command.bits.rows            := instruction.rs2Data(47, 32) +& 1.U
  command.ready                        := io.move.command.ready && !moving
  when(command.valid) {
    assert(
      instruction.raw_inst(6, 0) === "h7b".U &&
        instruction.raw_inst(31, 25) === MvoverISA.Funct.U && instruction.raw_inst(14, 12) === 3.U,
      "Move admission only accepts posted bank moves"
    )
  }
  when(io.move.command.fire)(moving    := true.B)
  io.move.completion.ready             := moving
  when(io.move.completion.fire)(moving := false.B)
  io.rocc.busy                         := moving
  io.rocc.interrupt                    := false.B
  io.rocc.resp.valid                   := false.B
  io.rocc.resp.bits                    := 0.U.asTypeOf(io.rocc.resp.bits)
}
