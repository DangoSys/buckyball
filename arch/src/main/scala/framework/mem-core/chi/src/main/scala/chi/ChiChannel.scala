package memcore.bus.chi

import chisel3._
import chisel3.util._

// Direction is from the transmitter. FLITV is a pulse, not ready/valid.
class ChiChannel(val flitBits: Int) extends Bundle {
  val flitpend = Output(Bool())
  val flitv    = Output(Bool())
  val flit     = Output(UInt(flitBits.W))
  val lcrdv    = Input(Bool())
}

// Active-link channel adapter. Link activation is provided by the endpoint.
// A coordinated reset is required to reclaim credits; runtime deactivation is not supported.
class ChiTx(flitBits: Int, maxCredits: Int = 15) extends Module {
  require(maxCredits >= 1 && maxCredits <= 15)

  val io = IO(new Bundle {
    val active  = Input(Bool())
    val in      = Flipped(Decoupled(UInt(flitBits.W)))
    val link    = new ChiChannel(flitBits)
    val credits = Output(UInt(log2Ceil(maxCredits + 1).W))
  })

  val credits  = RegInit(0.U(log2Ceil(maxCredits + 1).W))
  val returned = RegNext(io.link.lcrdv, false.B)
  io.in.ready := io.active && credits =/= 0.U
  val send = io.in.fire
  when(returned =/= send) {
    credits := Mux(returned, credits + 1.U, credits - 1.U)
  }
  when(returned && !send)(assert(credits < maxCredits.U, "CHI TX credit overflow"))
  when(send)(assert(credits =/= 0.U, "CHI TX credit underflow"))
  io.link.flitv := RegNext(send, false.B)
  io.link.flit     := RegEnable(io.in.bits, 0.U(flitBits.W), send)
  // Kept asserted throughout RUN, including the cycle before each FLITV.
  io.link.flitpend := io.active
  io.credits       := credits
  val wasActive = RegNext(io.active, false.B)
  when(wasActive)(assert(io.active, "CHI TX requires coordinated reset to stop"))
}

class ChiRx(flitBits: Int, depth: Int = 4) extends Module {
  require(depth >= 1 && depth <= 15)

  val io = IO(new Bundle {
    val active = Input(Bool())
    val link   = Flipped(new ChiChannel(flitBits))
    val out    = Decoupled(UInt(flitBits.W))
  })

  val queue      = Module(new Queue(UInt(flitBits.W), depth, pipe = true))
  val advertised = RegInit(0.U(log2Ceil(depth + 1).W))
  val grant      = io.active && (advertised +& queue.io.count) < depth.U
  io.link.lcrdv      := grant
  queue.io.enq.valid := io.link.flitv
  queue.io.enq.bits  := io.link.flit
  io.out <> queue.io.deq
  when(grant =/= io.link.flitv) {
    advertised := Mux(grant, advertised + 1.U, advertised - 1.U)
  }
  val previousPend = RegNext(io.link.flitpend, false.B)
  when(io.link.flitv) {
    assert(io.active, "CHI RX flit on inactive link")
    assert(previousPend, "CHI FLITPEND must precede FLITV")
    assert(advertised =/= 0.U, "CHI RX flit without granted credit")
    assert(queue.io.enq.ready, "CHI RX buffer overflow")
  }
  assert((advertised +& queue.io.count) <= depth.U, "CHI RX credit conservation")
  val wasActive = RegNext(io.active, false.B)
  when(wasActive)(assert(io.active, "CHI RX requires coordinated reset to stop"))
}
