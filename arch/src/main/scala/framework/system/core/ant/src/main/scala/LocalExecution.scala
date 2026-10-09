package framework.ant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.memory.{spm, tls}

@instantiable
class LocalExecution(p: Params) extends Module {

  @public val io = IO(new Bundle {
    val local      = new LocalPort(p)
    val npuDrained = Input(Bool())
    val command    = Decoupled(new Command(p))
    val response   = Flipped(Decoupled(new Response(p)))
  })

  val ant   = Instantiate(new Ant(p))
  val image = Instantiate(new tls.Store(spm.Params(0, p.codeBytes, p.data.dataBits), localReadOnly = true))
  val data  = Instantiate(new tls.Store(p.data))
  io.local.storageBusy := image.io.busy || data.io.busy
  ant.io.start.valid   := io.local.start.valid && io.local.inUse && !io.local.sharedHostBusy
  ant.io.start.bits    := io.local.start.bits
  io.local.start.ready := ant.io.start.ready && io.local.inUse && !io.local.sharedHostBusy
  io.local.result <> ant.io.result
  io.local.running     := ant.io.running
  io.local.retired     := ant.io.retired
  ant.io.cancel        := io.local.cancel
  ant.io.drained       := io.npuDrained && !image.io.busy && !data.io.busy && !io.local.sharedBusy
  io.command <> ant.io.command
  ant.io.response <> io.response
  val claimed = ant.io.running || (io.local.inUse && io.local.start.valid)
  image.io.running := claimed
  data.io.running  := claimed
  image.io.cancel  := io.local.cancel
  data.io.cancel   := io.local.cancel
  image.io.host <> io.local.code
  data.io.host <> io.local.data
  image.io.local <> ant.io.imem
  data.io.local <> ant.io.tls
  io.local.shared <> ant.io.tss
}
