package hier.tile.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.chi.RequesterPort
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.coherence.configs.CoherenceParams

/** CPU coherent Home and shared L2 with one backing-memory client. */
@instantiable
class Memory(p: CoherenceParams) extends Module {
  val c = p.chi

  @public
  val io = IO(new Bundle {
    val coherent        = Vec(p.agents, Flipped(new RequesterPort(c)))
    val backingReq      = Vec(1, Decoupled(new LineRequest(c)))
    val backingResp     = Vec(1, Flipped(Decoupled(new LineResponse(c))))
    val homeOutstanding = Output(UInt(log2Ceil(p.mshrEntries + 1).W))
  })

  val home: Instance[Home] = Instantiate(new Home(p))
  home.io.active            := true.B
  home.io.blockRequesterRsp := false.B
  home.io.requesters <> io.coherent
  io.backingReq(0) <> home.io.memoryReq
  home.io.memoryResp <> io.backingResp(0)
  io.homeOutstanding        := home.io.outstanding
}
