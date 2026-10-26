package framework.thread.ant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.thread.ant.configs.AntParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.thread.ant.{Completion, Executor, Retire, Start}
import framework.core.memory.access.preflight.PmpEntry
import framework.acc.frontend.RoCCIO
import framework.ip.memory.spm

/** Address space and permissions captured from the task's launch command. */
class TaskContext(permission: PmpParams) extends Bundle {
  val satp               = UInt(64.W)
  val effectivePrivilege = UInt(2.W)
  val sum                = Bool()
  val mxr                = Bool()
  val pmp                =
    Vec(permission.entries, new PmpEntry(permission))
}

/** One accepted launch binds the task context and starts its local program together. */
class Launch(b: AntParam, permission: PmpParams) extends Bundle {
  val descriptor = new Start(b)
  val context    = new TaskContext(permission)
}

/** Byte offsets into the TSS and the selected private store; transfers use 8-byte beats. */
class Transfer extends Bundle {
  val code         = Bool()
  val toLocal      = Bool()
  val localOffset  = UInt(32.W)
  val sharedOffset = UInt(32.W)
  val bytes        = UInt(32.W)
}

/** One thread task: launch, cancellation and completion. Storage accesses use independent interfaces. */
class Task(b: AntParam, permission: PmpParams) extends Bundle {
  val transfer    = Flipped(Decoupled(new Transfer))
  val transferred = Decoupled(Bool())
  val inUse       = Input(Bool())
  val start       = Flipped(Decoupled(new Launch(b, permission)))
  val result      = Decoupled(new Completion(b))
  val cancel      = Input(Bool())
  val running     = Output(Bool())
  val storageBusy = Output(Bool())

}

@instantiable
class Ant(val b: AntParam, permission: PmpParams) extends Module {

  @public
  val io = IO(new Bundle {
    val trace = Output(Valid(new Retire))
    val task  = new Task(b, permission)

    val shared = new spm.Port(b.shared)
    val rocc   = Flipped(new RoCCIO(64, permission))

  })

  val executor: Instance[Executor] = Instantiate(new Executor(b, permission))
  val image  = Instantiate(new LocalStore(b, 0))
  val data   = Instantiate(new LocalStore(b, 1))
  val loader = Instantiate(new Loader(b))

  io.trace                        := executor.io.retired
  io.task.storageBusy             := executor.io.storageBusy || loader.io.busy
  executor.io.start.valid         := io.task.start.valid && !loader.io.busy
  executor.io.start.bits          := io.task.start.bits
  io.task.start.ready             := executor.io.start.ready && !loader.io.busy
  loader.io.transfer.valid        := io.task.transfer.valid && !executor.io.storageClaimed && !io.task.start.valid
  loader.io.transfer.bits         := io.task.transfer.bits
  io.task.transfer.ready          := loader.io.transfer.ready && !executor.io.storageClaimed && !io.task.start.valid
  io.task.transferred <> loader.io.transferred
  io.task.result <> executor.io.result
  io.task.running                 := executor.io.running
  executor.io.cancel              := io.task.cancel
  executor.io.inUse               := io.task.inUse
  executor.io.localBusy(0)        := image.io.busy
  executor.io.localBusy(1)        := data.io.busy
  image.io.running                := executor.io.storageClaimed
  data.io.running                 := executor.io.storageClaimed
  image.io.cancel                 := io.task.cancel
  data.io.cancel                  := io.task.cancel
  image.io.host <> loader.io.code
  data.io.host <> loader.io.data
  image.io.local <> executor.io.imem
  data.io.local <> executor.io.tls
  io.shared.request.valid         := Mux(loader.io.busy, loader.io.shared.request.valid, executor.io.tss.request.valid)
  io.shared.request.bits          := Mux(loader.io.busy, loader.io.shared.request.bits, executor.io.tss.request.bits)
  loader.io.shared.request.ready  := loader.io.busy && io.shared.request.ready
  executor.io.tss.request.ready   := !loader.io.busy && io.shared.request.ready
  loader.io.shared.response.valid := loader.io.busy && io.shared.response.valid
  loader.io.shared.response.bits  := io.shared.response.bits
  executor.io.tss.response.valid  := !loader.io.busy && io.shared.response.valid
  executor.io.tss.response.bits   := io.shared.response.bits
  io.shared.response.ready        := Mux(loader.io.busy, loader.io.shared.response.ready, executor.io.tss.response.ready)
  io.rocc <> executor.io.rocc
}

/** Moves data between TSS and private storage without exposing private SRAM ports. */
@instantiable
class Loader(b: AntParam) extends Module {

  @public
  val io = IO(new Bundle {
    val transfer    = Flipped(Decoupled(new Transfer))
    val transferred = Decoupled(Bool())
    val busy        = Output(Bool())
    val code        = new spm.Port(spm.Params(0, b.codeBytes, b.data.dataBits))
    val data        = new spm.Port(b.data)
    val shared      = new spm.Port(b.shared)
  })

  val idle :: reading :: readResponse :: writing :: writeResponse :: completed :: Nil = Enum(6)
  val state                                                                           = RegInit(idle)
  val transfer                                                                        = Reg(new Transfer)
  val offset                                                                          = Reg(UInt(32.W))
  val value                                                                           = Reg(UInt(64.W))
  io.transfer.ready    := state === idle
  io.transferred.valid := state === completed
  io.transferred.bits  := true.B
  io.busy              := state =/= idle
  when(io.transfer.fire) {
    val t = io.transfer.bits
    assert(t.bytes =/= 0.U && !(t.localOffset | t.sharedOffset | t.bytes)(2, 0).orR)
    assert((t.localOffset +& t.bytes) <= Mux(t.code, b.codeBytes.U, b.data.bytes.U))
    assert((t.sharedOffset +& t.bytes) <= b.shared.bytes.U)
    transfer := t
    offset   := 0.U
    state    := reading
  }
  val local = Mux(transfer.code, 0.U, 1.U)
  val source      = Mux(transfer.toLocal, 2.U, local)
  val destination = Mux(transfer.toLocal, local, 2.U)
  val ports       = Seq(io.code, io.data, io.shared)
  for ((port, i) <- ports.zipWithIndex) {
    port.request.valid := (state === reading && source === i.U) || (state === writing && destination === i.U)
    port.request.bits  := 0.U.asTypeOf(port.request.bits)
    val base = Seq(BigInt(0), b.data.base, b.shared.base)(i)
    port.request.bits.address     := base.U + (if (i == 2) transfer.sharedOffset else transfer.localOffset) + offset
    port.request.bits.size        := 3.U
    port.request.bits.write       := state === writing
    port.request.bits.data        := value
    port.request.bits.mask        := Mux(state === writing, 255.U, 0.U)
    port.response.ready           := (state === readResponse && source === i.U) || (state === writeResponse && destination === i.U)
    when(port.request.fire)(state := Mux(state === reading, readResponse, writeResponse))
    when(port.response.fire) {
      assert(!port.response.bits.error, "Ant TSS transfer failed")
      when(state === readResponse) {
        value := port.response.bits.data(63, 0)
        state := writing
      }.otherwise {
        when(offset + 8.U === transfer.bytes)(state := completed)
          .otherwise { offset := offset + 8.U; state := reading }
      }
    }
  }
  when(io.transferred.fire)(state := idle)
}
