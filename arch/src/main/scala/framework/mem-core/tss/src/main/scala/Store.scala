package memcore.memory.tss

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.memory.spm.{Memory, Params => MemoryParams, Port, Response}

case class Params(memory: MemoryParams, ports: Int) {
  require(ports > 0)
}

/**
 * One shared SRAM, fair arbitration, one outstanding request and one response slot per port.
 * SRAM service is serialized; a held response does not block other clients. inUse covers the
 * lifetime of the cooperating group, not just one Ant. Cancel drains only its owning port.
 */
@instantiable
class Store(p: Params) extends Module {
  val count     = p.ports + 1
  val host      = p.ports
  val indexBits = log2Ceil(count)

  @public val io = IO(new Bundle {
    val inUse    = Input(Bool())
    val cancel   = Input(Vec(p.ports, Bool()))
    val clients  = Vec(p.ports, Flipped(new Port(p.memory)))
    val host     = Flipped(new Port(p.memory))
    val busy     = Output(Vec(p.ports, Bool()))
    val hostBusy = Output(Bool())
  })

  val memory    = Instantiate(new Memory(p.memory))
  val ports     = io.clients.toSeq :+ io.host
  val occupied  = RegInit(VecInit(Seq.fill(count)(false.B)))
  val available = RegInit(VecInit(Seq.fill(count)(false.B)))
  val cancelled = RegInit(VecInit(Seq.fill(count)(false.B)))
  val answers   = Reg(Vec(count, new Response(p.memory)))
  val active    = RegInit(false.B)
  val owner     = Reg(UInt(indexBits.W))
  val next      = RegInit(0.U(indexBits.W))
  val cancel    = VecInit(io.cancel.toSeq :+ false.B)

  val eligible = VecInit(ports.zipWithIndex.map { case (port, i) =>
    val allowed =
      if (i == host) !io.inUse && !occupied.take(p.ports).reduce(_ || _)
      else io.inUse && !occupied(host)
    port.request.valid && !occupied(i) && !cancel(i) && allowed && !reset.asBool
  })

  val after  = VecInit((0 until count).map(i => eligible(i) && i.U >= next)).asUInt
  val chosen = Mux(after.orR, PriorityEncoder(after), PriorityEncoder(eligible.asUInt))
  memory.io.readOnly            := false.B
  memory.io.port.request.valid  := !active && eligible.asUInt.orR
  memory.io.port.request.bits   := VecInit(ports.map(_.request.bits))(chosen)
  memory.io.port.response.ready := active
  when(memory.io.port.request.fire) {
    active            := true.B
    owner             := chosen
    occupied(chosen)  := true.B
    cancelled(chosen) := false.B
    next              := Mux(chosen === (count - 1).U, 0.U, chosen + 1.U)
  }
  for ((port, i) <- ports.zipWithIndex) {
    port.request.ready                          := !active && eligible(i) && chosen === i.U && memory.io.port.request.ready
    port.response.valid                         := available(i) && !cancelled(i) && !cancel(i) && !reset.asBool
    port.response.bits                          := answers(i)
    when(cancel(i) && occupied(i))(cancelled(i) := true.B)
    when(port.response.fire || (cancel(i) && available(i))) {
      available(i) := false.B
      occupied(i)  := false.B
      cancelled(i) := false.B
    }
  }
  when(memory.io.port.response.fire) {
    active := false.B
    when(cancelled(owner) || cancel(owner)) {
      occupied(owner)  := false.B
      cancelled(owner) := false.B
    }.otherwise {
      answers(owner)   := memory.io.port.response.bits
      available(owner) := true.B
    }
  }
  io.busy                       := VecInit(occupied.take(p.ports))
  io.hostBusy                   := occupied(host)
}
