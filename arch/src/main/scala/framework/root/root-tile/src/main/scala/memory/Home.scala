package hier.tile.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.chi._
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.coherence.Coherence
import memcore.memory.coherence.configs.CoherenceParams

/** One Tile-owned shared L2/Home and its explicit requester fabric. */
@instantiable
class Home(p: CoherenceParams) extends Module {
  val c = p.chi

  @public
  val io = IO(new Bundle {
    val active            = Input(Bool())
    val blockRequesterRsp = Input(Bool())
    val requesters        = Vec(p.agents, Flipped(new RequesterPort(c)))
    val memoryReq         = Decoupled(new LineRequest(c))
    val memoryResp        = Flipped(Decoupled(new LineResponse(c)))
    val outstanding       = Output(UInt(log2Ceil(p.mshrEntries + 1).W))
    val observedReq       = Output(Valid(new RequestFlit(c)))
    val observedRsp       = Output(Valid(new ResponseFlit(c)))
    val observedDat       = Output(Valid(new DataFlit(c)))
    val observedSnp       = Output(Valid(new DirectedSnoop(c)))
    val observedRxRsp     = Output(Valid(new ResponseFlit(c)))
    val observedRxDat     = Output(Valid(new DataFlit(c)))
  })

  val home:   Instance[Coherence] = Instantiate(new Coherence(p))
  val fabric: Instance[Fabric]    = Instantiate(new Fabric(c, p.agents, p.homeId))
  fabric.io.active            := io.active
  fabric.io.blockRequesterRsp := io.blockRequesterRsp
  fabric.io.requesters <> io.requesters
  home.io.req <> fabric.io.req
  home.io.rxRsp <> fabric.io.rxRsp
  home.io.rxDat <> fabric.io.rxDat
  fabric.io.rsp <> home.io.rsp
  fabric.io.dat <> home.io.dat
  fabric.io.snp <> home.io.snp
  io.memoryReq <> home.io.memoryReq
  home.io.memoryResp <> io.memoryResp
  io.outstanding              := home.io.outstanding
  io.observedReq.valid        := home.io.req.fire; io.observedReq.bits     := home.io.req.bits
  io.observedRsp.valid        := home.io.rsp.fire; io.observedRsp.bits     := home.io.rsp.bits
  io.observedDat.valid        := home.io.dat.fire; io.observedDat.bits     := home.io.dat.bits
  io.observedSnp.valid        := home.io.snp.fire; io.observedSnp.bits     := home.io.snp.bits
  io.observedRxRsp.valid      := home.io.rxRsp.fire; io.observedRxRsp.bits := home.io.rxRsp.bits
  io.observedRxDat.valid      := home.io.rxDat.fire; io.observedRxDat.bits := home.io.rxDat.bits
}
