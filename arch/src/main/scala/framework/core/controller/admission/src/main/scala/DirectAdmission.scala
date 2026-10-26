package framework.core.controller.admission

import framework.lib.queue.Queue
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.controller.configs.CommandsParam
import framework.acc.frontend.RoCCIO

/** Tracks posted accelerator commands and orders cross-core management after their drain. */
@instantiable
class DirectAdmission(p: CommandsParam) extends Module {
  require(p.remote)

  @public
  val io = IO(new Bundle {
    val execution   = Flipped(new Commands(p.tracking, p.dataBits, p.permission))
    val accelerator = Flipped(new RoCCIO(p.dataBits, p.permission))
    val remote      = new TaskEndpoint
    val drained     = Output(Bool())
  })

  val idle :: managing :: releasing :: Nil = Enum(3)
  val state                                = RegInit(idle)
  val tags                                 = Instantiate(new Queue(UInt(p.tracking.idBits.W), p.tracking.entries))
  val destination                          = Reg(UInt(5.W))
  val writeResult                          = Reg(Bool())
  val command                              = io.execution.command.bits
  val instruction                          = command.instruction
  val management                           = instruction.raw_inst(6, 0) === "h2b".U
  val available                            = state === idle && !io.remote.busy && !reset.asBool && tags.io.enq.ready &&
    (!management || (!io.accelerator.busy && !tags.io.deq.valid))

  io.accelerator.cmd.valid                   := io.execution.command.valid && available && !management
  io.accelerator.cmd.bits.instruction        := instruction
  io.accelerator.cmd.bits.satp               := command.satp
  io.accelerator.cmd.bits.effectivePrivilege := command.effectivePrivilege
  io.accelerator.cmd.bits.sum                := command.sum
  io.accelerator.cmd.bits.mxr                := command.mxr
  io.accelerator.cmd.bits.pmp                := command.pmp
  io.accelerator.resp.ready                  := false.B

  io.remote.request.valid          := io.execution.command.valid && available && management
  io.remote.request.bits.operation := instruction.raw_inst(31, 25)
  io.remote.request.bits.context   := instruction.rs1Data(63, 32)
  io.remote.request.bits.field     := instruction.rs1Data(31, 0)
  io.remote.request.bits.data      := instruction.rs2Data
  io.execution.command.ready       := available && Mux(management, io.remote.request.ready, io.accelerator.cmd.ready)
  when(io.execution.command.fire) {
    assert(
      Mux(
        management,
        instruction.raw_inst(31, 25) >= 13.U,
        instruction.raw_inst(6, 0) === "h7b".U && !instruction.raw_inst(14)
      ),
      "Direct core received an unsupported command"
    )
    destination            := instruction.raw_inst(11, 7)
    writeResult            := instruction.raw_inst(14)
    when(management)(state := managing)
  }
  tags.io.enq.valid                := io.execution.command.fire
  tags.io.enq.bits                 := command.tag
  io.execution.response.valid      := state === managing && writeResult && io.remote.reply.valid
  io.execution.response.bits.rd    := destination
  io.execution.response.bits.data  := io.remote.reply.bits
  io.remote.reply.ready            := state === managing && (!writeResult || io.execution.response.ready)
  when(io.remote.reply.fire)(state := releasing)
  val completed = state === releasing || (state === idle && !io.accelerator.busy)
  io.execution.complete.valid                                   := tags.io.deq.valid && completed
  io.execution.complete.bits.tag                                := tags.io.deq.bits
  tags.io.deq.ready                                             := io.execution.complete.ready && completed
  when(io.execution.complete.fire && state === releasing)(state := idle)
  io.drained                                                    := state === idle && !tags.io.deq.valid && !io.accelerator.busy && !io.remote.busy
  when(!reset.asBool) {
    assert(!io.accelerator.resp.valid, "Direct accelerator returned an unexpected register result")
    assert(!io.accelerator.interrupt, "Direct accelerator fault")
  }
}
