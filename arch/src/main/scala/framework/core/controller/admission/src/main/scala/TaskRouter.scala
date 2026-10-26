package framework.core.controller.admission

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.controller.configs.CommandsParam
import framework.device.cbus.ControlRequest
import framework.core.controller.task.TaskRequest
import framework.acc.frontend.Response

class TaskEndpoint extends Bundle {
  val request = Decoupled(new ControlRequest)
  val reply   = Flipped(Decoupled(UInt(64.W)))
  val busy    = Input(Bool())
}

/** Retains the reply destination and owner of each accepted management command. */
@instantiable
class TaskRouter(b: CommandsParam) extends Module {
  val remoteEnabled = b.remote

  @public
  val io = IO(new Bundle {

    val command = Flipped(Decoupled(new CommandSnapshot(
      b.tracking,
      b.dataBits,
      b.permission
    )))

    val response = Decoupled(new Response(b.dataBits))
    val busy     = Output(Bool())

    val local = new Bundle {
      val request = Decoupled(new TaskRequest(b.permission))
      val reply   = Flipped(Decoupled(UInt(64.W)))
      val busy    = Input(Bool())
    }

    val remote = Option.when(remoteEnabled)(new TaskEndpoint)
  })

  val pending     = RegInit(false.B)
  val remoteOwner = Reg(Bool())
  val destination = Reg(UInt(5.W))
  val command     = io.command.bits
  val instruction = command.instruction
  val remote      = instruction.raw_inst(31, 25) >= 13.U
  val request     = Wire(new ControlRequest)
  request.operation                                := instruction.raw_inst(31, 25)
  request.context                                  := instruction.rs1Data(63, 32)
  request.field                                    := instruction.rs1Data(31, 0)
  request.data                                     := instruction.rs2Data
  io.local.request.bits.control                    := request
  io.local.request.bits.context.satp               := command.satp
  io.local.request.bits.context.effectivePrivilege := command.effectivePrivilege
  io.local.request.bits.context.sum                := command.sum
  io.local.request.bits.context.mxr                := command.mxr
  io.local.request.bits.context.pmp                := command.pmp
  io.local.request.valid                           := io.command.valid && !pending && !remote
  io.local.reply.ready                             := pending && !remoteOwner && io.response.ready
  io.response.bits.rd                              := destination
  if (remoteEnabled) {
    val port = io.remote.get
    port.request.bits     := request
    port.request.valid    := io.command.valid && !pending && remote
    port.reply.ready      := pending && remoteOwner && io.response.ready
    io.command.ready      := !pending && Mux(remote, port.request.ready, io.local.request.ready)
    io.response.valid     := pending && Mux(remoteOwner, port.reply.valid, io.local.reply.valid)
    io.response.bits.data := Mux(remoteOwner, port.reply.bits, io.local.reply.bits)
    io.busy               := pending || io.local.busy || port.busy
  } else {
    io.command.ready      := !pending && !remote && io.local.request.ready
    io.response.valid     := pending && io.local.reply.valid
    io.response.bits.data := io.local.reply.bits
    io.busy               := pending || io.local.busy
    when(io.command.valid)(assert(!remote, "No cross-core task endpoint"))
  }
  when(io.command.fire) {
    pending     := true.B
    remoteOwner := remote
    destination := instruction.raw_inst(11, 7)
  }
  when(io.response.fire)(pending                   := false.B)
}
