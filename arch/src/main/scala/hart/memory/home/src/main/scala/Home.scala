package framework.hart.memory.home

import framework.hart.memory.home.configs.{Params => ModuleParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.bus.chi._
import framework.generic.bus.chi.snf.{LineRequest, LineResponse}
import framework.generic.memory.coherence.{Coherence, LineQuery, PageQuery}
import framework.generic.memory.coherence.configs.CoherenceParams

/**
 * One Hart-owned shared L2/Home and its explicit requester fabric; it answers `pages` DMA clients
 * whether a page has lines in its caches.
 */
@instantiable
class Home(b: ModuleParam, lineQuery: Boolean, pages: Int) extends Module {
  val p = b.memory

  val c = p.chi

  @public
  val io = IO(new Bundle {
    val active            = Input(Bool())
    val blockRequesterRsp = Input(Bool())

    val requesters = Vec(
      p.agents,
      Flipped(new RequesterPort(c))
    )

    val memoryReq = Decoupled(new LineRequest(c))

    val memoryResp = Flipped(Decoupled(new LineResponse(c)))

    val outstanding = Output(UInt(log2Ceil(p.mshrEntries + 1).W))

    val observedReq = Output(Valid(new RequestFlit(c)))

    val observedRsp = Output(Valid(new ResponseFlit(c)))

    val observedDat = Output(Valid(new DataFlit(c)))

    val observedSnp = Output(Valid(new DirectedSnoop(c)))

    val observedRxRsp = Output(Valid(new ResponseFlit(c)))

    val observedRxDat = Output(Valid(new DataFlit(c)))

  })

  @public
  val line = Option.when(lineQuery)(IO(new LineQuery(p)))
  @public
  val page = IO(Vec(pages, Flipped(new PageQuery(c.addressBits))))

  val home: Instance[Coherence] = Instantiate(new Coherence(b.memory, lineQuery, pages))
  line.foreach(_ <> home.line.get)
  page <> home.page

  val fabric: Instance[Fabric] = Instantiate(new Fabric(c, p.agents, p.homeId))

  fabric.io.active            := io.active
  fabric.io.blockRequesterRsp := io.blockRequesterRsp
  fabric.io.requesters <> io.requesters
  if (pages > 0) {
    val maintenance = Instantiate(new Maintenance(p))
    maintenance.io.request <> fabric.io.req
    home.io.req <> maintenance.io.req
    maintenance.io.rsp <> home.io.rsp
    fabric.io.rsp <> maintenance.io.response
    maintenance.io.query <> home.rangeLine.get
  } else {
    home.io.req <> fabric.io.req
    fabric.io.rsp <> home.io.rsp
  }
  home.io.rxRsp <> fabric.io.rxRsp
  home.io.rxDat <> fabric.io.rxDat
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
