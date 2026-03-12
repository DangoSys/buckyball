package framework.core.bbtile

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}

/**
 * BarrierUnit — tile-level hardware barrier for multi-core synchronization.
 *
 * Each core's BuckyballAccelerator raises arrive(i) after its ROB drains.
 * When all N cores have arrived, release(i) is asserted for one cycle,
 * allowing all cores to proceed simultaneously.
 *
 * @param nCores number of cores in this tile
 */
@instantiable
class BarrierUnit(val nCores: Int) extends Module {
  @public
  val io = IO(new Bundle {
    val arrive  = Input(Vec(nCores, Bool()))
    val release = Output(Vec(nCores, Bool()))
  })

  val arrived    = RegInit(VecInit(Seq.fill(nCores)(false.B)))
  val allArrived = arrived.asUInt.andR

  for (i <- 0 until nCores) {
    when(io.arrive(i)) { arrived(i) := true.B }
    io.release(i) := allArrived
  }

  when(allArrived) {
    for (i <- 0 until nCores) { arrived(i) := false.B }
  }
}
