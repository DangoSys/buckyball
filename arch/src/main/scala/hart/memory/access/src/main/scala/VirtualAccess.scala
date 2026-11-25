package framework.hart.memory.access

import framework.generic.contract.memory.access.UncachedRequest
import framework.generic.contract.memory.access.UncachedResponse

import buckyball.config.RegionHardware

import framework.hart.memory.access.configs.{Params => ModuleParam}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.bus.chi.rnf.{CacheAccess, CacheAtomic, CacheProbeResult, CacheResult}
import framework.hart.memory.access.mmu.Walker

class VirtualPort(b: ModuleParam) extends Bundle {
  val flushTranslation      = Input(Bool())
  val request               = Flipped(Decoupled(new VirtualRequest(b)))
  val response              = Decoupled(new VirtualResponse(b))
  val authorizationRequest  = Decoupled(new PhysicalAuthorization)
  val authorizationResponse = Flipped(Decoupled(Bool()))
}

/**
 * The pipelined D$ hit path. In the cycle a plain load or store is presented, it is translated by
 * the TLB, checked against PMP (`allowed`, from the CPU) and the regions, and enters the D$
 * (`accepted`); `complete` answers in the next cycle, where a store writes on `retire` and either
 * is dropped by `cancel`. Anything that does not enter or misses takes the VirtualPort.
 */
class DataProbe(b: ModuleParam) extends Bundle {
  val request    = Flipped(Valid(new VirtualRequest(b)))
  val paddr      = Output(UInt(64.W))
  val allowed    = Input(Bool())
  // The TLB, region and PMP pass it: it cannot fault, whether or not it entered the D$.
  val translated = Output(Bool())
  val accepted   = Output(Bool())
  val complete   = Valid(new CacheProbeResult)
  val cancel     = Input(Bool())
  val retire     = Input(Bool())
}

/** The TLB and region answer for a D$ probe; `hit` means it may enter the D$. */
class ProbeTranslation(b: ModuleParam) extends Bundle {
  val request = Input(Valid(new VirtualRequest(b)))
  val hit     = Output(Bool())
  val paddr   = Output(UInt(64.W))
}

class PhysicalAuthorization extends Bundle {
  val paddr     = UInt(64.W)
  val size      = UInt(3.W)
  val read      = Bool()
  val write     = Bool()
  val execute   = Bool()
  val isPte     = Bool()
  val privilege = UInt(2.W)
}

class VirtualRequest(b: ModuleParam) extends Bundle {
  val p = b.memory

  val vaddr     = UInt(64.W)
  val tag       = UInt(p.tagBits.W)
  val size      = UInt(3.W)
  val write     = Bool()
  val execute   = Bool()
  val signed    = Bool()
  val data      = UInt(64.W)
  val atomic    = UInt(4.W)
  val privilege = UInt(2.W)
  val sum       = Bool()
  val mxr       = Bool()
  val satpMode  = UInt(4.W)
  val rootPpn   = UInt(44.W)
}

class VirtualResponse(b: ModuleParam) extends Bundle {
  val p        = b.memory
  val lineBits = b.lineBits

  val tag         = UInt(p.tagBits.W)
  val data        = UInt(64.W)
  val misaligned  = Bool()
  val pageFault   = Bool()
  val accessFault = Bool()
  // Whole cached line of the translated word, for full-line clients only.
  val line        = UInt(lineBits.W)
}

/** Single-outstanding virtual data/instruction-word frontend; cache ownership is external. */
@instantiable
class VirtualAccess(b: ModuleParam, probes: Boolean) extends Module {
  val cp                 = b.memory
  val regions            = b.regions
  val lineBits           = b.lineBits
  val translationEntries = b.translationEntries

  require(regions.nonEmpty)
  for ((region, i) <- regions.zipWithIndex) {
    require(region.base >= 0 && region.bytes > 0 && region.base + region.bytes <= (BigInt(1) << cp.chi.addressBits))
    require(!region.cacheable || (region.base % 64 == 0 && region.bytes % 64 == 0))
    require(
      region.normal || (!region.cacheable && !region.executable && !region.atomic),
      "Device regions cannot be cached, executable or atomic"
    )
    for (other <- regions.drop(i + 1)) {
      require(region.base + region.bytes <= other.base || other.base + other.bytes <= region.base)
    }
  }

  @public
  val io = IO(new Bundle {
    val active           = Input(Bool())
    // SFENCE.VMA: drop every cached translation.
    val flushTranslation = Input(Bool())

    val request               = Flipped(Decoupled(new VirtualRequest(b)))
    val response              = Decoupled(new VirtualResponse(b))
    val authorizationRequest  = Decoupled(new PhysicalAuthorization)
    val authorizationResponse = Flipped(Decoupled(Bool()))

    val cacheRequest = Decoupled(new CacheAccess(cp.chi))

    val cacheResponse    = Flipped(Decoupled(new CacheResult(lineBits)))
    val uncachedRequest  = Decoupled(new UncachedRequest(b.memory))
    val uncachedResponse = Flipped(Decoupled(new UncachedResponse(b.memory)))

    val probe = Option.when(probes)(new ProbeTranslation(b))

    val observedTranslation =
      Output(Valid(new framework.hart.memory.access.mmu.Response(b.walker)))

    val observedPhysical = Output(Valid(new AccessRequest(b)))

    val observedCache = Output(Valid(new Bundle {
      val pte = Bool()

      val access = new CacheAccess(cp.chi)

    }))

  })

  val walker: Instance[Walker] = Instantiate(new Walker(b.walker))

  val physical: Instance[PhysicalAccess] = Instantiate(new PhysicalAccess(b))

  val idle :: startWalk :: waitWalk :: authorize :: waitAuthorization :: waitAccess :: respond :: Nil =
    Enum(7)
  val state                                                                                           = RegInit(idle)
  val command                                                                                         = Reg(new VirtualRequest(b))
  val answer                                                                                          = Reg(new VirtualResponse(b))
  val paddr                                                                                           = Reg(UInt(64.W))
  val cacheable                                                                                       = Reg(Bool())
  val normal                                                                                          = Reg(Bool())
  val pteInFlight                                                                                     = RegInit(false.B)
  val pteError                                                                                        = RegInit(false.B)
  val pteAuthorized                                                                                   = RegInit(false.B)
  val pteAuthorizationPending                                                                         = RegInit(false.B)
  val regionAllowed                                                                                   = Reg(Bool())
  def matches(first: UInt, last: UInt): Seq[Bool] =
    regions.map(region => first >= region.base.U(65.W) && last < (region.base + region.bytes).U(65.W))

  /** (cacheable, normal, permitted) of the regions an access of `r` at `address` touches. */
  def regionOf(address: UInt, r: VirtualRequest): (Bool, Bool, Bool) = {
    val last      = address.pad(65) + MuxLookup(r.size, 0.U(4.W))(Seq(1.U -> 1.U(4.W), 2.U -> 3.U(4.W), 3.U -> 7.U(4.W)))
    val hits      = matches(address.pad(65), last)
    val atomic    = r.atomic =/= CacheAtomic.None.U
    val reads     = !r.execute && !r.write && r.atomic =/= CacheAtomic.SC.U
    val writes    = r.write || (atomic && r.atomic =/= CacheAtomic.LR.U)
    val permitted = hits.zip(regions).map { case (hit, region) =>
      hit && (!region.normal || region.cacheable).B && (!r.execute || (region.normal && region.executable).B) &&
        (!reads || region.readable.B) && (!writes || region.writable.B) && (!atomic || region.atomic.B)
    }.reduce(_ || _)
    (
      hits.zip(regions).map { case (hit, region) => hit && region.cacheable.B }.reduce(_ || _),
      hits.zip(regions).map { case (hit, region) => hit && region.normal.B }.reduce(_ || _),
      permitted
    )
  }

  // A TLB hit translates in the cycle the request is accepted and presents its authorization then:
  // the access reaches the cache in the cycle the authorization answers.
  val incoming   = io.request.bits
  val misaligned = (incoming.vaddr & ((1.U(64.W) << incoming.size) - 1.U)).orR
  // A D$ probe and a VirtualPort request are never presented together; the probe uses the TLB then.
  val looked     = io.probe.map(probe => Mux(probe.request.valid, probe.request.bits, incoming)).getOrElse(incoming)
  walker.io.lookup.config.mode    := looked.satpMode
  walker.io.lookup.config.rootPpn := looked.rootPpn
  walker.io.lookup.req.vaddr      := looked.vaddr
  walker.io.lookup.req.write      := looked.write ||
    (looked.atomic =/= CacheAtomic.None.U && looked.atomic =/= CacheAtomic.LR.U)
  walker.io.lookup.req.execute    := looked.execute
  walker.io.lookup.req.privilege  := looked.privilege
  walker.io.lookup.req.sum        := looked.sum
  walker.io.lookup.req.mxr        := looked.mxr
  val fast                                       = walker.io.lookup.hit && !misaligned
  val (fastCacheable, fastNormal, fastPermitted) = regionOf(walker.io.lookup.paddr, incoming)
  io.request.ready := state === idle && io.active && (!fast || io.authorizationRequest.ready)
  io.probe.foreach { probe =>
    val r                                 = probe.request.bits
    val unaligned                         = (r.vaddr & ((1.U(64.W) << r.size) - 1.U)).orR
    val (probeCacheable, probeNormal, ok) = regionOf(walker.io.lookup.paddr, r)
    probe.hit   := walker.io.lookup.hit && !unaligned && probeCacheable && probeNormal && ok
    probe.paddr := walker.io.lookup.paddr
    when(probe.request.valid)(assert(!io.request.valid, "Virtual LSU probe overlaps a request"))
  }
  when(io.request.fire) {
    val r = io.request.bits
    assert(r.size <= 3.U, "Virtual LSU size must be 1, 2, 4 or 8 bytes")
    assert(r.atomic <= CacheAtomic.SC.U, "Virtual LSU does not accept Fence or unknown atomics")
    assert(
      r.atomic === CacheAtomic.None.U || (!r.write && r.size >= 2.U),
      "Virtual LSU atomic requires a 32 or 64 bit operand"
    )
    assert(r.satpMode === 0.U || r.satpMode === 8.U, "Virtual LSU requires Bare or Sv39 mode")
    assert(
      r.privilege === 0.U || r.privilege === 1.U || r.privilege === 3.U,
      "Virtual LSU effective privilege must be U, S or M"
    )
    assert(
      !r.execute || (!r.write && r.atomic === CacheAtomic.None.U && r.size >= 1.U && !r.signed),
      "Virtual instruction fetch requires unsigned 16, 32 or 64 bit read without atomic"
    )
    command           := r
    answer            := 0.U.asTypeOf(answer)
    answer.tag        := r.tag
    answer.misaligned := misaligned
    paddr             := walker.io.lookup.paddr
    cacheable         := fastCacheable
    normal            := fastNormal
    regionAllowed     := fastPermitted
    state             := Mux(misaligned, respond, Mux(fast, waitAuthorization, startWalk))
  }

  walker.io.flush                := io.flushTranslation
  walker.io.config.mode          := command.satpMode
  walker.io.config.rootPpn       := command.rootPpn
  walker.io.req.valid            := state === startWalk
  walker.io.req.bits.vaddr       := command.vaddr
  walker.io.req.bits.write       := command.write ||
    (command.atomic =/= CacheAtomic.None.U && command.atomic =/= CacheAtomic.LR.U)
  walker.io.req.bits.execute     := command.execute
  walker.io.req.bits.privilege   := command.privilege
  walker.io.req.bits.sum         := command.sum
  walker.io.req.bits.mxr         := command.mxr
  when(walker.io.req.fire)(state := waitWalk)
  walker.io.resp.ready           := state === waitWalk

  val (walkCacheable, walkNormal, permittedRegion) = regionOf(walker.io.resp.bits.paddr, command)
  val isAtomic                                     = command.atomic =/= CacheAtomic.None.U
  val readsData                                    = !command.execute && !command.write && command.atomic =/= CacheAtomic.SC.U
  val writesData                                   = command.write || (isAtomic && command.atomic =/= CacheAtomic.LR.U)

  when(walker.io.resp.fire) {
    when(walker.io.resp.bits.pageFault || walker.io.resp.bits.accessFault) {
      answer.pageFault   := walker.io.resp.bits.pageFault
      answer.accessFault := walker.io.resp.bits.accessFault
      state              := respond
    }.otherwise {
      paddr         := walker.io.resp.bits.paddr
      cacheable     := walkCacheable
      normal        := walkNormal
      regionAllowed := permittedRegion
      state         := authorize
    }
  }

  val authorized = state === waitAuthorization && io.authorizationResponse.valid && io.authorizationResponse.bits &&
    regionAllowed
  physical.io.request.valid            := authorized
  physical.io.request.bits.addr        := paddr
  physical.io.request.bits.tag         := command.tag
  physical.io.request.bits.size        := command.size
  physical.io.request.bits.write       := command.write
  physical.io.request.bits.signed      := command.signed
  physical.io.request.bits.data        := command.data
  physical.io.request.bits.atomic      := command.atomic
  physical.io.request.bits.cacheable   := cacheable
  physical.io.request.bits.normal      := normal
  when(physical.io.request.fire)(state := waitAccess)
  physical.io.response.ready           := state === waitAccess && io.response.ready
  when(physical.io.response.valid) {
    assert(physical.io.response.bits.tag === command.tag, "Virtual LSU physical response tag mismatch")
  }

  val walking    = state === startWalk || state === waitWalk
  val accessing  = state === waitAuthorization || state === waitAccess
  val pteFirst   = walker.io.access.bits.addr.pad(65)
  val pteRegions = matches(pteFirst, pteFirst + 7.U)

  val pteAllowed = pteRegions.zip(regions).map { case (hit, region) =>
    hit && (region.normal && region.readable && region.cacheable).B
  }.reduce(_ || _)

  // The Thread owns a per-command PMP snapshot. Each permission decision is a
  // single-outstanding handshake; neither a stalled query nor its response
  // allows speculative cache or MMIO traffic. PTE accesses use S privilege.
  val incomingAtomic = incoming.atomic =/= CacheAtomic.None.U
  io.authorizationRequest.valid          := state === authorize || (state === idle && io.active && io.request.valid && fast) ||
    (walking && walker.io.access.valid && !pteAuthorized && !pteAuthorizationPending && !pteError)
  io.authorizationRequest.bits.paddr     := Mux(walking, walker.io.access.bits.addr.pad(64), paddr)
  io.authorizationRequest.bits.size      := Mux(walking, 3.U, command.size)
  io.authorizationRequest.bits.read      := walking || readsData
  io.authorizationRequest.bits.write     := !walking && writesData
  io.authorizationRequest.bits.execute   := !walking && command.execute
  io.authorizationRequest.bits.isPte     := walking
  io.authorizationRequest.bits.privilege := Mux(walking, 1.U, command.privilege)
  when(state === idle) {
    io.authorizationRequest.bits.paddr     := walker.io.lookup.paddr.pad(64)
    io.authorizationRequest.bits.size      := incoming.size
    io.authorizationRequest.bits.read      := !incoming.execute && !incoming.write && incoming.atomic =/= CacheAtomic.SC.U
    io.authorizationRequest.bits.write     := incoming.write || (incomingAtomic && incoming.atomic =/= CacheAtomic.LR.U)
    io.authorizationRequest.bits.execute   := incoming.execute
    io.authorizationRequest.bits.privilege := incoming.privilege
  }
  when(io.authorizationRequest.fire) {
    when(walking) {
      pteAuthorizationPending := true.B
    }
      .otherwise(state := waitAuthorization)
  }
  io.authorizationResponse.ready         := pteAuthorizationPending ||
    (state === waitAuthorization && (!(io.authorizationResponse.bits && regionAllowed) || physical.io.request.ready))
  when(io.authorizationResponse.fire) {
    when(pteAuthorizationPending) {
      pteAuthorizationPending := false.B
      pteAuthorized           := true.B
      pteError                := !io.authorizationResponse.bits || !pteAllowed
    }.elsewhen(!(io.authorizationResponse.bits && regionAllowed)) {
      answer.accessFault := true.B; state := respond
    }
  }
  io.cacheRequest.valid                  := Mux(
    walking,
    walker.io.access.valid && pteAuthorized && !pteError,
    accessing && physical.io.cacheRequest.valid
  )
  io.cacheRequest.bits                   := physical.io.cacheRequest.bits
  when(walking) {
    io.cacheRequest.bits      := 0.U.asTypeOf(io.cacheRequest.bits)
    io.cacheRequest.bits.addr := walker.io.access.bits.addr
  }
  io.uncachedRequest.valid               := accessing && physical.io.uncachedRequest.valid
  io.uncachedRequest.bits                := physical.io.uncachedRequest.bits
  walker.io.access.ready                 := walking && pteAuthorized &&
    (pteError || io.cacheRequest.ready)
  when(walker.io.access.fire) {
    pteAuthorized := false.B
    pteInFlight   := !pteError
  }
  walker.io.result.valid                 := walking && ((!pteAuthorized && pteError) || (pteInFlight && io.cacheResponse.valid))
  walker.io.result.bits                  := 0.U.asTypeOf(walker.io.result.bits)
  walker.io.result.bits.data             := Mux(pteError || io.cacheResponse.bits.error, 0.U, io.cacheResponse.bits.data)
  walker.io.result.bits.error            := pteError || io.cacheResponse.bits.error
  when(walker.io.result.fire) { pteError := false.B; pteInFlight := false.B }
  physical.io.cacheRequest.ready         := accessing && io.cacheRequest.ready
  physical.io.cacheResponse.valid        := accessing && io.cacheResponse.valid
  physical.io.cacheResponse.bits         := io.cacheResponse.bits
  physical.io.uncachedRequest.ready      := accessing && io.uncachedRequest.ready
  physical.io.uncachedResponse.valid     := accessing && io.uncachedResponse.valid
  physical.io.uncachedResponse.bits      := io.uncachedResponse.bits
  io.cacheResponse.ready                 := Mux(
    walking,
    pteInFlight && walker.io.result.ready,
    accessing && physical.io.cacheResponse.ready
  )
  io.uncachedResponse.ready              := accessing && physical.io.uncachedResponse.ready
  when(io.cacheResponse.valid) {
    assert((walking && pteInFlight) || accessing, "Virtual LSU cache result has no owner")
  }
  when(io.uncachedResponse.valid) {
    assert(accessing, "Virtual LSU uncached result has no owner")
  }

  io.response.valid            := state === respond || (state === waitAccess && physical.io.response.valid)
  io.response.bits             := answer
  when(state === waitAccess) {
    io.response.bits.data                   := physical.io.response.bits.data
    io.response.bits.misaligned             := physical.io.response.bits.misaligned
    io.response.bits.accessFault            := physical.io.response.bits.accessFault
    if (lineBits > 0) io.response.bits.line := physical.io.response.bits.line
  }
  when(io.response.fire)(state := idle)
  io.observedTranslation.valid := walker.io.resp.fire || (io.request.fire && fast)
  io.observedTranslation.bits  := walker.io.resp.bits
  when(state === idle) {
    io.observedTranslation.bits       := 0.U.asTypeOf(io.observedTranslation.bits)
    io.observedTranslation.bits.paddr := walker.io.lookup.paddr
  }
  io.observedPhysical.valid    := physical.io.request.fire
  io.observedPhysical.bits     := physical.io.request.bits
  io.observedCache.valid       := io.cacheRequest.fire
  io.observedCache.bits.pte    := walking
  io.observedCache.bits.access := io.cacheRequest.bits
}
