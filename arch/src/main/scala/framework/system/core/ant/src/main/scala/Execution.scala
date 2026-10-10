package framework.ant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.memory.{spm, tss}

/** A tile-local group. The controller holds inUse until all cooperating tasks have finished. */
@instantiable
class Execution(p: Params, contexts: Int) extends Module {
  require(contexts > 0)
  val codeParams = spm.Params(0, p.codeBytes, p.data.dataBits)

  @public val io = IO(new Bundle {
    val inUse       = Input(Bool())
    val start       = Vec(contexts, Flipped(Decoupled(new Start(p))))
    val result      = Vec(contexts, Decoupled(new Completion(p)))
    val cancel      = Input(Vec(contexts, Bool()))
    val npuDrained  = Input(Vec(contexts, Bool()))
    val running     = Output(Vec(contexts, Bool()))
    val storageBusy = Output(Bool())
    val command     = Vec(contexts, Decoupled(new Command(p)))
    val response    = Vec(contexts, Flipped(Decoupled(new Response(p))))
    val retired     = Output(Vec(contexts, Valid(new Retire)))
    val code        = Vec(contexts, Flipped(new spm.Port(codeParams)))
    val data        = Vec(contexts, Flipped(new spm.Port(p.data)))
    val shared      = Flipped(new spm.Port(p.shared))
  })

  val shared = Instantiate(new tss.Store(tss.Params(p.shared, contexts)))
  // A premature group release cannot grant management access over a running task.
  shared.io.inUse  := io.inUse || io.running.asUInt.orR
  shared.io.host <> io.shared
  shared.io.cancel := io.cancel
  val privateBusy = Wire(Vec(contexts, Bool()))
  io.storageBusy := privateBusy.asUInt.orR || shared.io.busy.asUInt.orR || shared.io.hostBusy
  for (i <- 0 until contexts) {
    val execution = Instantiate(new LocalExecution(p))
    privateBusy(i)                    := execution.io.local.storageBusy
    execution.io.local.inUse          := io.inUse
    execution.io.local.start <> io.start(i)
    io.result(i) <> execution.io.local.result
    io.running(i)                     := execution.io.local.running
    io.retired(i)                     := execution.io.local.retired
    execution.io.local.cancel         := io.cancel(i)
    execution.io.local.sharedBusy     := shared.io.busy(i)
    execution.io.local.sharedHostBusy := shared.io.hostBusy
    execution.io.npuDrained           := io.npuDrained(i)
    io.command(i) <> execution.io.command
    execution.io.response <> io.response(i)
    execution.io.local.code <> io.code(i)
    execution.io.local.data <> io.data(i)
    shared.io.clients(i) <> execution.io.local.shared
  }
}
