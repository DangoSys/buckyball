package memcore.bus.chi

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class TxIO(flitBits: Int, creditBits: Int) extends Bundle {
  val active  = Input(Bool())
  val in      = Flipped(Decoupled(UInt(flitBits.W)))
  val link    = new Channel(flitBits)
  val credits = Output(UInt(creditBits.W))
}

// Active-link channel adapter. Link activation is provided by the endpoint.
// A coordinated reset is required to reclaim credits; runtime deactivation is not supported.
@instantiable
class Tx(flitBits: Int, maxCredits: Int = 15) extends Module {
  require(flitBits >= 1)
  require(maxCredits >= 1 && maxCredits <= 15)
  val creditBits = log2Ceil(maxCredits + 1)

  @public
  val io = IO(new TxIO(flitBits, creditBits))

  // Credit state
  val credits  = RegInit(0.U(creditBits.W))
  val returned = RegNext(io.link.lcrdv, false.B)
  val send     = io.in.fire

  when(returned =/= send) {
    credits := Mux(returned, credits + 1.U, credits - 1.U)
  }

  // Internal and physical channel interfaces
  io.in.ready      := io.active && credits =/= 0.U
  io.link.flitpend := io.active
  io.link.flitv    := RegNext(send, false.B)
  io.link.flit     := RegEnable(io.in.bits, 0.U(flitBits.W), send)
  io.credits       := credits

  // Link contract checks
  val wasActive = RegNext(io.active, false.B)
  when(returned && !send)(assert(credits < maxCredits.U, "CHI TX credit overflow"))
  when(send)(assert(credits =/= 0.U, "CHI TX credit underflow"))
  when(wasActive)(assert(io.active, "CHI TX requires coordinated reset to stop"))
}
