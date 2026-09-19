package memcore.bus.axi

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class PacketArbiterIO(
  inputs:   Int,
  dataBits: Int,
  idBits:   Int,
  destBits: Int,
  userBits: Int)
    extends Bundle {
  val in  = Vec(inputs, Flipped(Decoupled(new Beat(dataBits, idBits, destBits, userBits))))
  val out = Decoupled(new Beat(dataBits, idBits, destBits, userBits))
}

/** A packet-preserving AXI-S arbiter. Arbitration changes only after last. */
@instantiable
class PacketArbiter(
  inputs:   Int,
  dataBits: Int,
  idBits:   Int = 0,
  destBits: Int = 0,
  userBits: Int = 0)
    extends Module {
  require(inputs >= 1)
  @public
  val io = IO(new PacketArbiterIO(inputs, dataBits, idBits, destBits, userBits))

  val choiceBits  = math.max(1, log2Ceil(inputs))
  val locked      = RegInit(false.B)
  val selected    = RegInit(0.U(choiceBits.W))
  val validInputs = VecInit(io.in.map(_.valid))
  val arbitration = PriorityEncoder(validInputs)
  val active      = Mux(locked, selected, arbitration)
  val hasInput    = Mux(locked, io.in(selected).valid, validInputs.asUInt.orR)

  io.out.valid := hasInput
  io.out.bits  := io.in(active).bits
  for (i <- 0 until inputs) {
    io.in(i).ready := io.out.ready && io.out.valid && active === i.U
  }

  when(io.out.fire) {
    locked   := !io.out.bits.last
    selected := active
  }
}
