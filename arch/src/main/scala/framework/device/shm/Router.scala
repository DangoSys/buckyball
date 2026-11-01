package framework.device.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.ip.bus.axi4

/**
 * A circuit-switched router of the AXI writes between communication nodes.
 *
 * `routes(input)(output)` lists the target cores (the address bits above `localBits`) that `input` sends to
 * `output`; an empty list is no connection. A write locks its output from its AW to its B, so a burst crosses the
 * router whole and an output carries one write at a time. The links carry no reads.
 */
@instantiable
class Router(p: axi4.Params, localBits: Int, routes: Seq[Seq[Seq[Int]]]) extends Module {
  val inputs    = routes.size
  val outputs   = routes.head.size
  require(routes.forall(_.size == outputs))
  val inputBits = math.max(1, log2Ceil(inputs))

  @public
  val io = IO(new Bundle {
    val in  = Vec(inputs, Flipped(new axi4.Port(p)))
    val out = Vec(outputs, new axi4.Port(p))
  })

  val locked = RegInit(VecInit(Seq.fill(outputs)(false.B)))
  val owner  = Reg(Vec(outputs, UInt(inputBits.W)))
  val awSent = Reg(Vec(outputs, Bool()))
  val wDone  = Reg(Vec(outputs, Bool()))
  val last   = RegInit(VecInit(Seq.fill(outputs)(0.U(inputBits.W))))

  // An input holding an output waits for its B before it asks for another.
  val held = VecInit((0 until inputs).map(i => (0 until outputs).map(o => locked(o) && owner(o) === i.U).reduce(_ || _)))

  for (i <- 0 until inputs) {
    val connected = (0 until outputs).filter(o => routes(i)(o).nonEmpty)
    def mine(o: Int) = locked(o) && owner(o) === i.U
    io.in(i).aw.ready := connected.map(o => mine(o) && !awSent(o) && io.out(o).aw.ready).foldLeft(false.B)(_ || _)
    io.in(i).w.ready  := connected.map(o => mine(o) && awSent(o) && !wDone(o) && io.out(o).w.ready)
      .foldLeft(false.B)(_ || _)
    io.in(i).b.valid  := connected.map(o => mine(o) && io.out(o).b.valid).foldLeft(false.B)(_ || _)
    io.in(i).b.bits   := Mux1H(connected.map(o => mine(o) -> io.out(o).b.bits))
    io.in(i).ar.ready := false.B
    io.in(i).r.valid  := false.B
    io.in(i).r.bits   := DontCare
    when(io.in(i).ar.valid)(assert(false.B, "Device links carry no reads"))
  }

  for (o <- 0 until outputs) {
    val sources = (0 until inputs).filter(i => routes(i)(o).nonEmpty)
    val request = VecInit((0 until inputs).map { i =>
      val target = io.in(i).aw.bits.addr >> localBits
      io.in(i).aw.valid && !held(i) && routes(i)(o).map(target === _.U).foldLeft(false.B)(_ || _)
    })
    // Round robin over the inputs, starting after the last owner.
    val after   = VecInit((0 until inputs).map(i => request(i) && i.U > last(o)))
    val choice  = Mux(after.asUInt.orR, PriorityEncoder(after), PriorityEncoder(request))
    when(!locked(o) && request.asUInt.orR) {
      locked(o) := true.B
      owner(o)  := choice
      last(o)   := choice
      awSent(o) := false.B
      wDone(o)  := false.B
    }

    val port = io.out(o)
    def from(i:                 Int)            = owner(o) === i.U
    def owned[T <: Data](field: axi4.Port => T) = Mux1H(sources.map(i => from(i) -> field(io.in(i))))
    port.aw.valid := locked(o) && !awSent(o) && owned(_.aw.valid)
    port.aw.bits  := owned(_.aw.bits)
    port.w.valid  := locked(o) && awSent(o) && !wDone(o) && owned(_.w.valid)
    port.w.bits   := owned(_.w.bits)
    port.b.ready  := locked(o) && owned(_.b.ready)
    port.ar.valid := false.B
    port.ar.bits  := DontCare
    port.r.ready  := false.B

    when(port.aw.fire)(awSent(o)                   := true.B)
    when(port.w.fire && port.w.bits.last)(wDone(o) := true.B)
    when(port.b.fire)(locked(o)                    := false.B)
  }
}
