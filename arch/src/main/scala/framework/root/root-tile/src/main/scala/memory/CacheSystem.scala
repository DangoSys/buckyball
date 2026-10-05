package hier.tile.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.chi._
import memcore.bus.chi.rnf.{BankedChiCache, CacheAccess, CacheResult, RnfParams}
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.coherence.configs.CoherenceParams

/** Independent canonical cache composition with physical active-link CHI channels. */
@instantiable
class CacheSystem(p: CoherenceParams) extends Module {
  require(p.agents == 2 && p.homeId == 64 && p.mshrEntries == 4)
  private val c = p.chi

  @public
  val io = IO(new Bundle {
    val active            = Input(Bool())
    val blockRequesterRsp = Input(Bool())
    val access            = Vec(p.agents, Flipped(Decoupled(new CacheAccess(c))))
    val result            = Vec(p.agents, Decoupled(new CacheResult))
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

  val home: Instance[Home] = Instantiate(new Home(p))

  val cores: Seq[Instance[BankedChiCache]] = Seq.tabulate(p.agents)(i =>
    Instantiate(new BankedChiCache(
      RnfParams(c, nodeId = i + 1, cacheLines = 8, homeId = p.homeId, homeCount = 1, banks = 2)
    ))
  )

  home.io.active            := io.active
  home.io.blockRequesterRsp := io.blockRequesterRsp
  io.memoryReq <> home.io.memoryReq
  home.io.memoryResp <> io.memoryResp
  io.outstanding            := home.io.outstanding
  for (i <- 0 until p.agents) {
    cores(i).io.access <> io.access(i)
    io.result(i) <> cores(i).io.result
    home.io.requesters(i) <> cores(i).io.chi
  }
  io.observedReq   := home.io.observedReq
  io.observedRsp   := home.io.observedRsp
  io.observedDat   := home.io.observedDat
  io.observedSnp   := home.io.observedSnp
  io.observedRxRsp := home.io.observedRxRsp
  io.observedRxDat := home.io.observedRxDat

}
