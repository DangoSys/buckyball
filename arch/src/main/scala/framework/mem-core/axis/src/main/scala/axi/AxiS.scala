package memcore.bus.axi

import chisel3._
import chisel3.util._

/**
 * AXI4-Stream payload signals.
 *
 * A `Decoupled[AxiSBeat]` is the full AXI-S channel: `valid` and `ready`
 * are TVALID and TREADY. This bundle exposes the remaining standard signal
 * names directly in generated RTL.
 */
class AxiSBeat(
  dataBits: Int,
  idBits:   Int = 0,
  destBits: Int = 0,
  userBits: Int = 0)
    extends Bundle {
  require(dataBits > 0 && dataBits % 8 == 0)
  val tdata = UInt(dataBits.W)
  val tkeep = UInt((dataBits / 8).W)
  val tlast = Bool()
  val tid   = UInt(idBits.W)
  val tdest = UInt(destBits.W)
  val tuser = UInt(userBits.W)

  // Internal aliases preserve concise datapath code while retaining AXI-S names.
  def data: UInt = tdata
  def keep: UInt = tkeep
  def last: Bool = tlast
  def id:   UInt = tid
  def dest: UInt = tdest
  def user: UInt = tuser
}

/** A packet-preserving AXI-S arbiter. Arbitration changes only after last. */
class AxiSPacketArbiter(
  inputs:   Int,
  dataBits: Int,
  idBits:   Int = 0,
  destBits: Int = 0,
  userBits: Int = 0)
    extends Module {
  require(inputs >= 1)
  private val beat = new AxiSBeat(dataBits, idBits, destBits, userBits)

  val io = IO(new Bundle {
    val in  = Vec(inputs, Flipped(Decoupled(beat)))
    val out = Decoupled(beat)
  })

  private val choiceBits = math.max(1, log2Ceil(inputs))
  val locked             = RegInit(false.B)
  val selected           = RegInit(0.U(choiceBits.W))
  val validInputs        = VecInit(io.in.map(_.valid))
  val arbitration        = PriorityEncoder(validInputs)
  val active             = Mux(locked, selected, arbitration)
  val hasInput           = Mux(locked, io.in(selected).valid, validInputs.asUInt.orR)

  io.out.valid := hasInput
  io.out.bits  := io.in(active).bits
  for (i <- 0 until inputs) {
    io.in(i).ready := io.out.ready && io.out.valid && active === i.U
  }

  when(io.out.fire) {
    when(locked && io.out.bits.last) {
      locked := false.B
    }.elsewhen(!locked && !io.out.bits.last) {
      locked   := true.B
      selected := active
    }
  }
}

object EmitAxiSPacketArbiter extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new AxiSPacketArbiter(inputs = 2, dataBits = 32),
    firtoolOpts = args.drop(1) ++ Seq("--split-verilog", "-o=build"),
    args = Array("--target-dir", "build")
  )
}
