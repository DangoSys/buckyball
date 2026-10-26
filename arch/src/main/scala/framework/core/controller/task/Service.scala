package framework.core.controller.task

import chisel3.experimental.hierarchy.Instance

import framework.core.controller.configs.TasksParam
import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.core.tss.TssControlPort
import framework.thread.ant.Task
import framework.ip.memory.spm

/**
 * Core execution service. The caller supplies the uncached management transport and
 * task-bound NPU/DMA owner; neither CPU cache nor address translation lives here.
 */
@instantiable
class Service(val b: TasksParam) extends Module {
  val contexts = b.count

  @public
  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new TaskRequest(b.permission)))
    val reply   = Decoupled(UInt(64.W))

    val tasks = MixedVec(b.locals.map(p => Flipped(new Task(p, b.permission))))

    val idle = Output(Bool())
    val tss  = new TssControlPort(b.tss)
  })

  val control: Instance[Control] = Instantiate(new Control(b))

  require(b.tss.ports == contexts)

  val transferBusy = Wire(Bool())
  val running      = VecInit(io.tasks.map(_.running))
  val storageBusy  = io.tasks.map(_.storageBusy).reduce(_ || _) ||
    io.tss.busy.asUInt.orR || io.tss.hostBusy
  control.io.storageBusy := storageBusy
  io.tss.inUse           := control.io.inUse || running.asUInt.orR || transferBusy
  io.tss.cancel          := control.io.cancel
  for (i <- 0 until contexts) {
    io.tasks(i).inUse          := control.io.inUse
    io.tasks(i).start.valid    := control.io.start(i).valid && !io.tss.hostBusy && !io.tss.busy(i)
    io.tasks(i).start.bits     := control.io.start(i).bits
    control.io.start(i).ready  := io.tasks(i).start.ready && !io.tss.hostBusy && !io.tss.busy(i)
    // A cancelled Ant may finish locally before its accepted TSS access drains.
    control.io.result(i).valid := io.tasks(i).result.valid && !io.tss.busy(i)
    control.io.result(i).bits  := io.tasks(i).result.bits
    io.tasks(i).result.ready   := control.io.result(i).ready && !io.tss.busy(i)
    io.tasks(i).cancel         := control.io.cancel(i)
  }
  val idle :: controlReply :: transferring :: sharedResponse :: loadReply :: Nil = Enum(5)
  val state      = RegInit(idle)
  val request    = io.request.bits.control
  val isTransfer = request.operation >= 7.U && request.operation <= 10.U
  val isShared   = request.operation === 11.U || request.operation === 12.U
  val isLoad     = isTransfer || isShared
  val localIndex = request.context(math.max(1, log2Ceil(contexts)) - 1, 0)
  val owner      = Reg(UInt(math.max(1, log2Ceil(contexts)).W))
  val value      = Reg(UInt(64.W))
  transferBusy := state === transferring
  val transferAllowed = request.context < contexts.U && control.io.available(localIndex) &&
    !running(localIndex) && !VecInit(io.tasks.map(_.storageBusy))(localIndex) && !io.tss.busy(localIndex)
  for (i <- 0 until contexts) {
    val task = io.tasks(i)
    task.transfer.valid             := state === idle && io.request.valid && isTransfer && transferAllowed && request.context === i.U
    task.transfer.bits.code         := request.operation <= 8.U
    task.transfer.bits.toLocal      := !request.operation(0)
    task.transfer.bits.localOffset  := request.field
    task.transfer.bits.sharedOffset := request.data(31, 0)
    task.transfer.bits.bytes        := request.data(63, 32)
    task.transferred.ready          := state === transferring && owner === i.U
    when(task.transferred.fire) {
      assert(task.transferred.bits, "Ant TSS transfer failed")
      value := 0.U
      state := loadReply
    }
  }
  val sharedValid = request.context === 0.U && request.field(2, 0) === 0.U &&
    (request.field +& 8.U) <= b.tss.memory.bytes.U
  val port = io.tss.host
  port.request.valid        := state === idle && io.request.valid && isShared && sharedValid && !reset.asBool
  port.request.bits         := 0.U.asTypeOf(port.request.bits)
  port.request.bits.address := b.tss.memory.base.U + request.field
  port.request.bits.size    := 3.U
  port.request.bits.write   := request.operation === 12.U
  port.request.bits.data    := request.data
  port.request.bits.mask    := Mux(request.operation === 12.U, 255.U, 0.U)
  port.response.ready       := state === sharedResponse
  when(port.response.fire) {
    assert(!port.response.bits.error, "TSS management access failed")
    value := port.response.bits.data(63, 0)
    state := loadReply
  }
  when(state === idle && io.request.valid && isLoad && !reset.asBool) {
    assert(Mux(isTransfer, transferAllowed, sharedValid), "Invalid TSS transfer or storage ownership")
  }
  control.io.request.valid  := state === idle && io.request.valid && !isLoad && !reset.asBool
  control.io.request.bits   := io.request.bits
  io.request.ready          := state === idle && !reset.asBool && Mux(
    isTransfer,
    transferAllowed && VecInit(io.tasks.map(_.transfer.ready))(localIndex),
    Mux(isShared, sharedValid && port.request.ready, control.io.request.ready)
  )
  when(io.request.fire) {
    owner := localIndex
    state := Mux(isTransfer, transferring, Mux(isShared, sharedResponse, controlReply))
  }
  io.reply.valid            := state === loadReply || (state === controlReply && control.io.response.valid)
  io.reply.bits             := Mux(state === loadReply, value, control.io.response.bits)
  control.io.response.ready := state === controlReply && io.reply.ready
  when(io.reply.fire)(state := idle)
  io.idle                   := state === idle && !running.asUInt.orR && !storageBusy
}
