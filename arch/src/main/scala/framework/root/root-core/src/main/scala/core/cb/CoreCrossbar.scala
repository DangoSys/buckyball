package hier.core.cb

import chisel3._
import chisel3.util._
import memcore.bus.axi.{AxiSBeat, AxiSPacketArbiter}

/**
 * Core-local AXI-S crossbar.
 *
 * TDEST selects an egress. Each egress arbitrates whole packets, so a source
 * retains its selected target from the first beat through TLAST.
 */
class CoreCrossbar(
  initiators: Int,
  targets:    Int,
  dataBits:   Int,
  idBits:     Int = 0,
  destBits:   Int = 4,
  userBits:   Int = 0)
    extends Module {
  require(initiators >= 1 && targets >= 1)
  require(destBits >= log2Ceil(targets))
  private val beat = new AxiSBeat(dataBits, idBits, destBits, userBits)

  val io = IO(new Bundle {
    val in  = Vec(initiators, Flipped(Decoupled(beat)))
    val out = Vec(targets, Decoupled(beat))
  })

  val arbiters = Seq.fill(targets)(Module(new AxiSPacketArbiter(initiators, dataBits, idBits, destBits, userBits)))
  for (target <- 0 until targets) {
    io.out(target) <> arbiters(target).io.out
    for (source <- 0 until initiators) {
      arbiters(target).io.in(source).valid := io.in(source).valid && io.in(source).bits.tdest === target.U
      arbiters(target).io.in(source).bits  := io.in(source).bits
    }
  }
  for (source <- 0 until initiators) {
    val selected = io.in(source).bits.tdest
    io.in(source).ready := MuxLookup(selected, false.B)(
      arbiters.zipWithIndex.map { case (arbiter, target) => target.U -> arbiter.io.in(source).ready }
    )
    when(io.in(source).valid) {
      assert(selected < targets.U, "Core crossbar TDEST has no local target")
    }
  }
}
