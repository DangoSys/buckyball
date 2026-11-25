package framework.hart.memory

import framework.generic.contract.memory.access.UncachedRequest
import framework.generic.contract.memory.access.UncachedResponse

import framework.generic.arbiter.{Arbiter}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.memory.configs.MemoryParam
import framework.hart.memory.uncached.Uncached
import framework.hart.memory.access.{DataProbe, VirtualAccess, VirtualPort}
import framework.hart.memory.access.cache.Cache
import framework.hart.memory.home.Home
import framework.generic.bus.axi4
import framework.generic.bus.chi.RequesterPort
import framework.generic.bus.chi.snf.{LineRequest, LineResponse}
import framework.generic.bus.chi.rnf.{BankedChiCache, CacheAccess, CacheAtomic}
import framework.generic.queue.Queue

/** Translation, cache services and ordered cached/uncached reply routing. */
@instantiable
class MemorySystem(b: MemoryParam, externalHome: Boolean) extends Module {
  require(b.uncached.ports == 1, "MemorySystem arbitrates requests into one uncached adapter input")
  val hart   = b
  val access = hart.memoryAccess.memory
  require(b.icache.chi == access.chi, "Instruction cache and memory access CHI must match")
  require(b.dcache.cache.chi == access.chi, "Data cache and memory access CHI must match")
  if (!externalHome) require(b.home.memory.chi == access.chi, "Local Home and cache fabric CHI must match")

  @public
  val io = IO(new Bundle {
    val fetch          = new VirtualPort(b.memoryAccess)
    val memory         = new VirtualPort(b.memoryAccess.copy(lineBits = 0))
    val probe          = Option.when(b.dcache.cache.probe)(new DataProbe(b.memoryAccess.copy(lineBits = 0)))
    val timeRequest    = Flipped(Decoupled(new UncachedRequest(b.memoryAccess.memory)))
    val timeResponse   = Decoupled(new UncachedResponse(b.memoryAccess.memory))
    val services       = Flipped(new MemCtrl(b.dcache.tracking, b.memoryAccess.walker))
    val commandAllowed = Input(Bool())
    val uncached       = new axi4.Port(b.uncached.axi)

    val coherent = Vec(if (externalHome) 2 else 0, new RequesterPort(access.chi))

    val backing = Option.when(!externalHome)(new Bundle {
      val request  = Decoupled(new LineRequest(hart.home.memory.chi))
      val response = Flipped(Decoupled(new LineResponse(hart.home.memory.chi)))
    })

  })

  val fetch: Instance[VirtualAccess] =
    Instantiate(new VirtualAccess(b.memoryAccess, false))

  val memory: Instance[VirtualAccess] =
    Instantiate(new VirtualAccess(b.memoryAccess.copy(lineBits = 0), b.dcache.cache.probe))

  val icache: Instance[BankedChiCache] = Instantiate(new BankedChiCache(hart.icache))

  val dcache: Instance[Cache] =
    Instantiate(new Cache(b.dcache, externalHome))

  fetch.io.active            := true.B
  fetch.io.flushTranslation  := io.fetch.flushTranslation
  fetch.io.request <> io.fetch.request
  io.fetch.response <> fetch.io.response
  io.fetch.authorizationRequest <> fetch.io.authorizationRequest
  fetch.io.authorizationResponse <> io.fetch.authorizationResponse
  memory.io.active           := true.B
  memory.io.flushTranslation := io.memory.flushTranslation
  memory.io.request <> io.memory.request
  memory.io.request.valid    := io.memory.request.valid && io.commandAllowed
  io.memory.request.ready    := memory.io.request.ready && io.commandAllowed
  io.memory.response <> memory.io.response
  io.memory.authorizationRequest <> memory.io.authorizationRequest
  memory.io.authorizationResponse <> io.memory.authorizationResponse
  val services      = io.services
  val memoryAllowed = services.cpuAllow
  // A D$ probe that may enter the D$ this cycle; the interlock answers it before any other access.
  val probing       = io.probe.map(probe => probe.request.valid && memory.io.probe.get.hit && probe.allowed).getOrElse(false.B)
  val probePaddr    = io.probe.map(_ => memory.io.probe.get.paddr & ~7.U(64.W)).getOrElse(0.U(64.W))
  services.cpuQuery          := 0.U.asTypeOf(services.cpuQuery)
  services.cpuQuery.valid    := probing || memory.io.cacheRequest.valid || fetch.io.cacheRequest.valid
  services.cpuQuery.paddr    := Mux(
    probing,
    probePaddr,
    Mux(memory.io.cacheRequest.valid, memory.io.cacheRequest.bits.addr, fetch.io.cacheRequest.bits.addr)
  )
  services.cpuQuery.sizeLog2 := Mux(
    !probing && memory.io.cacheRequest.valid &&
      memory.io.cacheRequest.bits.atomic =/= CacheAtomic.None.U && memory.io.cacheRequest.bits.atomicWord,
    2.U,
    3.U
  )
  services.cpuQuery.write    := Mux(
    probing,
    io.probe.map(_.request.bits.write).getOrElse(false.B),
    memory.io.cacheRequest.valid && (memory.io.cacheRequest.bits.write ||
      (memory.io.cacheRequest.bits.atomic =/= CacheAtomic.None.U && memory.io.cacheRequest.bits.atomic =/= CacheAtomic.LR.U))
  )

  icache.io.access.valid      := fetch.io.cacheRequest.valid && memoryAllowed && !probing
  icache.io.access.bits       := fetch.io.cacheRequest.bits
  fetch.io.cacheRequest.ready := icache.io.access.ready && memoryAllowed && !probing
  fetch.io.cacheResponse <> icache.io.result

  val cacheRequests = Instantiate(new Arbiter(
    new CacheAccess(access.chi),
    2
  ))

  cacheRequests.io.in(0).valid     := memory.io.cacheRequest.valid && memoryAllowed && !probing
  cacheRequests.io.in(0).bits      := memory.io.cacheRequest.bits
  memory.io.cacheRequest.ready     := cacheRequests.io.in(0).ready && memoryAllowed && !probing
  cacheRequests.io.in(1).valid     := services.pteRequest.valid
  cacheRequests.io.in(1).bits      := 0.U.asTypeOf(cacheRequests.io.in(1).bits)
  cacheRequests.io.in(1).bits.addr := services.pteRequest.bits.addr
  services.pteRequest.ready        := cacheRequests.io.in(1).ready

  val cacheOwners = Instantiate(new Queue(Bool(), 2, false, false))

  dcache.io.access.valid          := cacheRequests.io.out.valid && cacheOwners.io.enq.ready
  dcache.io.access.bits           := cacheRequests.io.out.bits
  cacheRequests.io.out.ready      := dcache.io.access.ready && cacheOwners.io.enq.ready
  cacheOwners.io.enq.valid        := dcache.io.access.fire
  cacheOwners.io.enq.bits         := cacheRequests.io.chosen === 1.U
  memory.io.cacheResponse.valid   := dcache.io.result.valid && cacheOwners.io.deq.valid && !cacheOwners.io.deq.bits
  memory.io.cacheResponse.bits    := dcache.io.result.bits
  services.pteResponse.valid      := dcache.io.result.valid && cacheOwners.io.deq.valid && cacheOwners.io.deq.bits
  services.pteResponse.bits.data  := dcache.io.result.bits.data
  services.pteResponse.bits.error := dcache.io.result.bits.error
  dcache.io.result.ready          := cacheOwners.io.deq.valid && Mux(
    cacheOwners.io.deq.bits,
    services.pteResponse.ready,
    memory.io.cacheResponse.ready
  )
  cacheOwners.io.deq.ready        := dcache.io.result.fire
  dcache.io.maintenance <> services.maintenance
  io.probe.foreach { probe =>
    val translation = memory.io.probe.get
    val cache       = dcache.io.probe.get
    val r           = probe.request.bits
    val mask        = MuxLookup(r.size, 255.U(8.W))(Seq(0.U -> 1.U(8.W), 1.U -> 3.U(8.W), 2.U -> 15.U(8.W)))
    translation.request  := probe.request
    probe.paddr          := translation.paddr
    probe.translated     := probing
    cache.req.valid      := probing && memoryAllowed
    cache.req.bits.addr  := translation.paddr(access.chi.addressBits - 1, 0)
    cache.req.bits.write := r.write
    cache.req.bits.data  := r.data << (translation.paddr(2, 0) << 3)
    cache.req.bits.mask  := Mux(r.write, mask << translation.paddr(2, 0), 0.U)
    probe.accepted       := cache.req.fire
    probe.complete.valid := cache.complete.valid
    probe.complete.bits  := cache.complete.bits
    cache.complete.ready := true.B
    cache.cancel         := probe.cancel
    cache.retire         := probe.retire
  }
  services.maintained <> dcache.io.maintained
  dcache.io.olderRequestsDrained  := cacheOwners.io.count === 0.U

  val uncachedRequests = Instantiate(new Arbiter(
    new UncachedRequest(b.memoryAccess.memory),
    3
  ))

  uncachedRequests.io.in(0) <> fetch.io.uncachedRequest
  uncachedRequests.io.in(1) <> memory.io.uncachedRequest

  val uncached: Instance[Uncached] = Instantiate(new Uncached(b.memoryAccess.memory, b.uncached))

  uncached.io.request(0) <> uncachedRequests.io.out
  io.uncached <> uncached.io.axi
  uncachedRequests.io.in(2) <> io.timeRequest
  val uncachedOwner = Reg(UInt(2.W))
  when(uncachedRequests.io.out.fire)(uncachedOwner := uncachedRequests.io.chosen)
  fetch.io.uncachedResponse.valid                  := uncached.io.response(0).valid && uncachedOwner === 0.U
  memory.io.uncachedResponse.valid                 := uncached.io.response(0).valid && uncachedOwner === 1.U
  fetch.io.uncachedResponse.bits                   := uncached.io.response(0).bits
  memory.io.uncachedResponse.bits                  := uncached.io.response(0).bits

  val timeResponse = io.timeResponse

  timeResponse.valid            := uncached.io.response(0).valid && uncachedOwner === 2.U
  timeResponse.bits             := uncached.io.response(0).bits
  uncached.io.response(0).ready := VecInit(Seq(
    fetch.io.uncachedResponse.ready,
    memory.io.uncachedResponse.ready,
    timeResponse.ready
  ))(uncachedOwner)
  services.drained              := dcache.io.drained
  if (externalHome) {
    // Another hart's cache sits behind the external Home, so every line goes there.
    services.lineMaintain := true.B
    io.coherent(0) <> icache.io.chi
    io.coherent(1) <> dcache.io.chi
  } else {
    val home: Instance[Home] = Instantiate(new Home(b.home, true, 0))
    home.line.get.address     := services.lineAddress
    home.line.get.write       := services.lineWrite
    services.lineMaintain     := home.line.get.maintain
    home.io.active            := true.B
    home.io.blockRequesterRsp := false.B
    home.io.requesters(0) <> icache.io.chi
    home.io.requesters(1) <> dcache.io.chi
    home.io.memoryReq <> io.backing.get.request
    home.io.memoryResp <> io.backing.get.response
  }

}
