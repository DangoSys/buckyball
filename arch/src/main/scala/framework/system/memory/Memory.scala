package framework.system.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.ddr.{Params => DdrParams, Bridge}

/** CPU coherence backing traffic and NPU DMA meet at the DDR AXI arbiter. */
@instantiable
class Memory(ddr: DdrParams, dmaMasters: Int = 0) extends Module {

  @public
  val io = IO(new Bundle {
    val lineRequest  = Vec(ddr.clients, Flipped(Decoupled(new LineRequest(ddr.line))))
    val lineResponse = Vec(ddr.clients, Decoupled(new LineResponse(ddr.line)))
    val dma          = Vec(dmaMasters, Flipped(new memcore.bus.axi4.Port(ddr.axi)))
    val axi          = new memcore.bus.axi4.Port(ddr.axi)
    val outstanding  = Output(UInt(log2Ceil(ddr.slots + 2 * (1 << ddr.axi.idBits) + 1).W))
  })

  val bridge: Instance[Bridge]                        = Instantiate(new Bridge(ddr))
  bridge.io.request <> io.lineRequest
  io.lineResponse <> bridge.io.response
  val fabric: Instance[memcore.bus.axi4.Interconnect] =
    Instantiate(new memcore.bus.axi4.Interconnect(ddr.axi, dmaMasters + 1))
  fabric.io.in(0) <> bridge.io.axi
  io.axi <> fabric.io.out
  for (i <- 0 until dmaMasters) {
    fabric.io.in(i + 1) <> io.dma(i)
  }
  io.outstanding := bridge.io.outstanding +& fabric.io.outstanding
}
