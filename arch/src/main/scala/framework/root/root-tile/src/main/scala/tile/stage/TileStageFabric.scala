package hier.tile.stage

import chisel3._
import chisel3.util._
import memcore.bus.axi.AxiSBeat

/**
 * Directed fabric between heterogeneous homogeneous-core groups inside a tile.
 *
 * Every forward link is an AXI-S packet channel from stage N to N+1. Reverse
 * control is separate, so data ownership and completion/backpressure do not
 * require a tile-local ring.
 */
class TileStageFabric(stages: Int, dataBits: Int, controlBits: Int = 32) extends Module {
  require(stages >= 2 && controlBits > 0)
  private val data = new AxiSBeat(dataBits)

  val io = IO(new Bundle {
    val forwardIn  = Vec(stages - 1, Flipped(Decoupled(data)))
    val forwardOut = Vec(stages - 1, Decoupled(data))
    val reverseIn  = Vec(stages - 1, Flipped(Decoupled(UInt(controlBits.W))))
    val reverseOut = Vec(stages - 1, Decoupled(UInt(controlBits.W)))
  })

  for (link <- 0 until stages - 1) {
    io.forwardOut(link) <> io.forwardIn(link)
    io.reverseOut(link) <> io.reverseIn(link)
  }
}
