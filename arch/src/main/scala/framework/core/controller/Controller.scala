package framework.core.controller

import framework.thread.ant.Task
import framework.ip.memory.spm
import chisel3._
import chisel3.util.MixedVec
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.controller.configs.ControllerParam
import framework.core.controller.admission.{CommandAdmission, Commands, TaskEndpoint, TaskRouter}
import framework.core.controller.task.Service
import framework.acc.memdomain.MvoverPort
import framework.core.tss.TssControlPort

/** Core command admission, task management and thread launch control. */
@instantiable
class Controller(b: ControllerParam) extends Module {
  val commands = b.commands

  @public
  val io = IO(new Bundle {

    val execution     = Flipped(new Commands(commands.tracking, commands.dataBits, commands.permission))
    val threads       = b.tasks.map(p => MixedVec(p.locals.map(local => Flipped(new Task(local, p.permission)))))
    val move          = new MvoverPort
    val remote        = Option.when(commands.remote)(new TaskEndpoint)
    val dataflow      = Option.when(b.tasks.nonEmpty)(new TaskEndpoint)
    val tss           = b.tasks.map(p => new TssControlPort(p.tss))
    val memoryAllowed = Output(Bool())
    val drained       = Output(Bool())
  })

  val admission: Instance[CommandAdmission] = Instantiate(new CommandAdmission(commands))
  val router:    Instance[TaskRouter]       = Instantiate(new TaskRouter(commands))
  admission.io.execution <> io.execution
  admission.io.move <> io.move
  if (b.tasks.nonEmpty) {
    val input       = admission.io.taskRequest
    val operation   = input.bits.instruction.raw_inst(31, 25)
    val selected    = operation >= 20.U && operation <= 23.U
    val flow        = io.dataflow.get
    val pending     = RegInit(false.B)
    val destination = Reg(UInt(5.W))
    flow.request.valid              := input.valid && selected
    flow.request.bits.operation     := operation
    flow.request.bits.context       := input.bits.instruction.rs1Data(63, 32)
    flow.request.bits.field         := input.bits.instruction.rs1Data(31, 0)
    flow.request.bits.data          := input.bits.instruction.rs2Data
    router.io.command.valid         := input.valid && !selected
    router.io.command.bits          := input.bits
    input.ready                     := Mux(selected, flow.request.ready, router.io.command.ready)
    when(flow.request.fire) { pending := true.B; destination := input.bits.instruction.raw_inst(11, 7) }
    admission.io.taskResponse.valid := Mux(pending, flow.reply.valid, router.io.response.valid)
    admission.io.taskResponse.bits  := router.io.response.bits
    when(pending) {
      admission.io.taskResponse.bits.rd   := destination
      admission.io.taskResponse.bits.data := flow.reply.bits
    }
    flow.reply.ready                := pending && admission.io.taskResponse.ready
    router.io.response.ready        := !pending && admission.io.taskResponse.ready
    when(flow.reply.fire)(pending   := false.B)
  } else {
    router.io.command <> admission.io.taskRequest
    admission.io.taskResponse <> router.io.response
  }
  io.remote.zip(router.io.remote).foreach { case (port, endpoint) => port <> endpoint }
  io.memoryAllowed := admission.io.memoryAllowed
  io.drained := admission.io.idle && !router.io.busy

  if (b.tasks.nonEmpty) {
    val service: Instance[Service] = Instantiate(new Service(b.tasks.get))
    service.io.request <> router.io.local.request
    router.io.local.reply <> service.io.reply
    router.io.local.busy := !service.io.idle
    io.tss.get <> service.io.tss
    service.io.tasks <> io.threads.get
  } else {
    router.io.local.request.ready := false.B
    router.io.local.reply.valid   := false.B
    router.io.local.reply.bits    := 0.U
    router.io.local.busy          := false.B
  }
}
