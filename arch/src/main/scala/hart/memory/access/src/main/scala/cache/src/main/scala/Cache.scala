package framework.hart.memory.access.cache

import framework.generic.arbiter.{RRArbiter}

import framework.hart.memory.access.cache.configs.{Params => ModuleParam}

import framework.generic.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.bus.chi.{HomeMapping, RequestFlit, RequesterPort}
import framework.generic.bus.chi.rnf.{BankedChiCache, CacheAccess, CacheProbe, CacheResult, RnfParams}
import framework.generic.memory.interlock.{Acknowledgement, Maintenance => Range, Params}

/** The CPU's L1 and range maintenance client share one explicit CHI requester. */
@instantiable
class Cache(b: ModuleParam, rangeHome: Boolean) extends Module {
  val config   = b.cache
  val tracking = b.tracking

  val c = config.chi

  @public
  val io = IO(new Bundle {

    val access = Flipped(Decoupled(new CacheAccess(c)))

    val result =
      Decoupled(new CacheResult(0))

    val probe = Option.when(config.probe)(new CacheProbe(c))

    val maintenance          = Flipped(Decoupled(new Range(b.tracking)))
    val maintained           = Decoupled(new Acknowledgement(b.tracking))
    // Includes previously accepted virtual accesses, page walks and fetches outside this L1.
    val olderRequestsDrained = Input(Bool())
    val drained              = Output(Bool())

    val chi = new RequesterPort(c)

    val outstanding = Output(UInt(32.W))
  })

  val cache: Instance[BankedChiCache] = Instantiate(new BankedChiCache(b.cache))

  val maintenance: Instance[Maintenance] = Instantiate(new Maintenance(
    c,
    config.nodeId,
    HomeMapping(config.homeCount, config.homeId),
    config.banks,
    tracking,
    rangeHome
  ))

  val drained = io.olderRequestsDrained && cache.io.outstanding === 0.U
  io.drained                   := drained
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
  io.probe.foreach { probe =>
    probe <> cache.io.probe.get
    cache.io.probe.get.req.valid := probe.req.valid && !admitMaintenance
    probe.req.ready              := cache.io.probe.get.req.ready && !admitMaintenance
  }
  maintenance.io.drained := true.B
  io.maintained <> maintenance.io.response
  io.outstanding         := cache.io.outstanding

  val requests = Instantiate(new RRArbiter(
    new RequestFlit(c),
    2
  ))

  requests.io.in(0) <> cache.io.chi.req
  requests.io.in(1) <> maintenance.io.req

  val requestQueue = Instantiate(new Queue(new RequestFlit(c), 2, false, false))
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
