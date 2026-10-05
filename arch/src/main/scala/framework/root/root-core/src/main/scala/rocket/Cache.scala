package hier.core.rocket

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.chi.{RequestFlit, RequesterPort}
import memcore.bus.chi.rnf.{BankedChiCache, CacheAccess, CacheProbe, CacheResult, RnfParams}
import memcore.memory.interlock.{Acknowledgement, Maintenance => Range, Params}

/** A core's own L1 and range maintenance client share one explicit CHI requester. */
@instantiable
class Cache(config: RnfParams, tracking: Params) extends Module {
  override def desiredName: String = "CoreCache"
  private val c = config.chi

  @public
  val io = IO(new Bundle {
    val access               = Flipped(Decoupled(new CacheAccess(c)))
    val result               = Decoupled(new CacheResult)
    val probe                = Option.when(config.probe)(new CacheProbe(c))
    val maintenance          = Flipped(Decoupled(new Range(tracking)))
    val maintained           = Decoupled(new Acknowledgement(tracking))
    // Includes previously accepted virtual accesses, page walks and fetches outside this L1.
    val olderRequestsDrained = Input(Bool())
    val chi                  = new RequesterPort(c)
    val outstanding          = Output(UInt(32.W))
  })

  val cache:       Instance[BankedChiCache] = Instantiate(new BankedChiCache(config))
  val maintenance: Instance[Maintenance]    = Instantiate(new Maintenance(config, tracking))
  val drained = io.olderRequestsDrained && cache.io.outstanding === 0.U
  maintenance.io.request.valid := io.maintenance.valid && drained
  maintenance.io.request.bits  := io.maintenance.bits
  io.maintenance.ready         := maintenance.io.request.ready && drained
  // Existing owners may keep progressing until drained. At admission, a new CPU
  // request cannot race the CMO; subsequent conflicts belong to the external interlock.
  val admitMaintenance = io.maintenance.valid && io.maintenance.ready
  cache.io.access.valid  := io.access.valid && !admitMaintenance
  cache.io.access.bits   := io.access.bits
  io.access.ready        := cache.io.access.ready && !admitMaintenance
  io.result <> cache.io.result
  io.probe.foreach(_ <> cache.io.probe.get)
  maintenance.io.drained := true.B
  io.maintained <> maintenance.io.response
  io.outstanding         := cache.io.outstanding

  val requests = Module(new RRArbiter(new RequestFlit(c), 2) {

    override lazy val lastGrant = {
      val previous = RegInit(0.U(1.W))
      when(io.out.fire)(previous := io.chosen)
      previous
    }

  })

  requests.io.in(0) <> cache.io.chi.req
  requests.io.in(1) <> maintenance.io.req
  val requestQueue = Module(new Queue(new RequestFlit(c), 2))
  requestQueue.io.enq <> requests.io.out
  io.chi.req <> requestQueue.io.deq
  io.chi.txRsp <> cache.io.chi.txRsp
  io.chi.txDat <> cache.io.chi.txDat
  cache.io.chi.rxDat <> io.chi.rxDat
  cache.io.chi.snp <> io.chi.snp

  val maintenanceResponse = io.chi.rxRsp.bits.txnId === config.banks.U
  maintenance.io.rsp.valid := io.chi.rxRsp.valid && maintenanceResponse
  maintenance.io.rsp.bits  := io.chi.rxRsp.bits
  cache.io.chi.rxRsp.valid := io.chi.rxRsp.valid && !maintenanceResponse
  cache.io.chi.rxRsp.bits  := io.chi.rxRsp.bits
  io.chi.rxRsp.ready       := Mux(maintenanceResponse, maintenance.io.rsp.ready, cache.io.chi.rxRsp.ready)
}
