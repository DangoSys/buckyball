package memcore.memory.tls

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.memory.spm.{Memory, Params, Port}

/**
 * A context-private local store. Loader and execution access it in separate ownership phases.
 * Cancellation drains an accepted local access and discards its response before reuse.
 */
@instantiable
class Store(p: Params, localReadOnly: Boolean = false) extends Module {

  @public val io = IO(new Bundle {
    val running = Input(Bool())
    val cancel  = Input(Bool())
    val host    = Flipped(new Port(p))
    val local   = Flipped(new Port(p))
    val busy    = Output(Bool())
  })

  val memory    = Instantiate(new Memory(p))
  val busy      = RegInit(false.B)
  val owner     = Reg(Bool())
  val cancelled = RegInit(false.B)
  val discard   = owner && (cancelled || io.cancel)
  memory.io.port.request.valid               := !busy && !reset.asBool &&
    Mux(io.running, io.local.request.valid && !io.cancel, io.host.request.valid)
  memory.io.port.request.bits                := Mux(io.running, io.local.request.bits, io.host.request.bits)
  memory.io.readOnly                         := io.running && localReadOnly.B
  io.local.request.ready                     := !busy && io.running && !io.cancel && memory.io.port.request.ready
  io.host.request.ready                      := !busy && !io.running && memory.io.port.request.ready
  when(memory.io.port.request.fire) {
    busy      := true.B
    owner     := io.running
    cancelled := false.B
  }
  when(busy && owner && io.cancel)(cancelled := true.B)
  io.local.response.valid                    := busy && owner && !discard && memory.io.port.response.valid
  io.host.response.valid                     := busy && !owner && memory.io.port.response.valid
  io.local.response.bits                     := memory.io.port.response.bits
  io.host.response.bits                      := memory.io.port.response.bits
  memory.io.port.response.ready              := busy &&
    (discard || Mux(owner, io.local.response.ready, io.host.response.ready))
  when(memory.io.port.response.fire) { busy := false.B; cancelled := false.B }
  io.busy                                    := busy
}
