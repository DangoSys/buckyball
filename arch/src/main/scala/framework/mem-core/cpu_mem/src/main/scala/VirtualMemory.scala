package memcore.memory.cpu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.chi.rnf.{CacheAccess, CacheAtomic, CacheResult}
import memcore.memory.mmu.Walker

case class PhysicalRegion(
  base:       BigInt,
  bytes:      BigInt,
  cacheable:  Boolean,
  executable: Boolean,
  readable:   Boolean,
  writable:   Boolean,
  atomic:     Boolean,
  normal:     Boolean)

class PhysicalAuthorization extends Bundle {
  val paddr     = UInt(64.W)
  val size      = UInt(3.W)
  val read      = Bool()
  val write     = Bool()
  val execute   = Bool()
  val isPte     = Bool()
  val privilege = UInt(2.W)
}

class VirtualMemoryRequest(p: CpuMemParams) extends Bundle {
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

class VirtualMemoryResponse(p: CpuMemParams, lineBits: Int = 0) extends Bundle {
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
class VirtualMemory(
  cp:                 CpuMemParams,
  regions:            Seq[PhysicalRegion],
  lineBits:           Int = 0,
  translationEntries: Int = 0)
    extends Module {
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

    // Combinational translation through the TLB only, for a caller's fast path.
    val lookup = new Bundle {
      val config = Input(new memcore.memory.mmu.Config)
      val req    = Input(new memcore.memory.mmu.Request)
      val hit    = Output(Bool())
      val paddr  = Output(UInt(cp.chi.addressBits.W))
    }

    val request               = Flipped(Decoupled(new VirtualMemoryRequest(cp)))
    val response              = Decoupled(new VirtualMemoryResponse(cp, lineBits))
    val authorizationRequest  = Decoupled(new PhysicalAuthorization)
    val authorizationResponse = Flipped(Decoupled(Bool()))
    val cacheRequest          = Decoupled(new CacheAccess(cp.chi))
    val cacheResponse         = Flipped(Decoupled(new CacheResult(lineBits)))
    val uncachedRequest       = Decoupled(new UncachedRequest(cp))
    val uncachedResponse      = Flipped(Decoupled(new UncachedResponse(cp)))
    val observedTranslation   = Output(Valid(new memcore.memory.mmu.Response(cp.chi)))
    val observedPhysical      = Output(Valid(new CpuMemoryRequest(cp)))

    val observedCache = Output(Valid(new Bundle {
      val pte    = Bool()
      val access = new CacheAccess(cp.chi)
    }))

  })

  val walker:   Instance[Walker] = Instantiate(new Walker(cp.chi, translationEntries))
  val physical: Instance[CpuMem] = Instantiate(new CpuMem(cp, lineBits))

  val idle :: startWalk :: waitWalk :: authorize :: waitAuthorization :: startAccess :: waitAccess :: respond :: Nil =
    Enum(8)
  val state                                                                                                          = RegInit(idle)
  val command                                                                                                        = Reg(new VirtualMemoryRequest(cp))
  val answer                                                                                                         = Reg(new VirtualMemoryResponse(cp, lineBits))
  val paddr                                                                                                          = Reg(UInt(64.W))
  val cacheable                                                                                                      = Reg(Bool())
  val normal                                                                                                         = Reg(Bool())
  val pteInFlight                                                                                                    = RegInit(false.B)
  val pteError                                                                                                       = RegInit(false.B)
  val pteAuthorized                                                                                                  = RegInit(false.B)
  val pteAuthorizationPending                                                                                        = RegInit(false.B)
  val regionAllowed                                                                                                  = Reg(Bool())
  def matches(first: UInt, last: UInt): Seq[Bool] =
    regions.map(region => first >= region.base.U(65.W) && last < (region.base + region.bytes).U(65.W))
  io.request.ready := state === idle && io.active
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
      !r.execute || (!r.write && r.atomic === CacheAtomic.None.U && r.size === 3.U && !r.signed),
      "Virtual instruction word requires unsigned 64 bit read without atomic"
    )
    command    := r
    answer     := 0.U.asTypeOf(answer)
    answer.tag := r.tag
    val misaligned = (r.vaddr & ((1.U(64.W) << r.size) - 1.U)).orR
    answer.misaligned := misaligned
    state             := Mux(misaligned, respond, startWalk)
  }

  walker.io.flush                := io.flushTranslation
  walker.io.lookup.config        := io.lookup.config
  walker.io.lookup.req           := io.lookup.req
  io.lookup.hit                  := walker.io.lookup.hit
  io.lookup.paddr                := walker.io.lookup.paddr
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

  val lastByte = walker.io.resp.bits.paddr.pad(65) + MuxLookup(command.size, 0.U(4.W))(
    Seq(1.U -> 1.U(4.W), 2.U -> 3.U(4.W), 3.U -> 7.U(4.W))
  )

  val dataRegions = matches(walker.io.resp.bits.paddr.pad(65), lastByte)
  val isAtomic    = command.atomic =/= CacheAtomic.None.U
  val readsData   = !command.execute && !command.write && command.atomic =/= CacheAtomic.SC.U
  val writesData  = command.write || (isAtomic && command.atomic =/= CacheAtomic.LR.U)

  val permittedRegion = dataRegions.zip(regions).map { case (hit, region) =>
    hit && (!region.normal || region.cacheable).B && (!command.execute || (region.normal && region.executable).B) &&
      (!readsData || region.readable.B) && (!writesData || region.writable.B) &&
      (!isAtomic || region.atomic.B)
  }.reduce(_ || _)

  when(walker.io.resp.fire) {
    when(walker.io.resp.bits.pageFault || walker.io.resp.bits.accessFault) {
      answer.pageFault   := walker.io.resp.bits.pageFault
      answer.accessFault := walker.io.resp.bits.accessFault
      state              := respond
    }.otherwise {
      paddr         := walker.io.resp.bits.paddr
      cacheable     := dataRegions.zip(regions).map { case (hit, region) => hit && region.cacheable.B }.reduce(_ || _)
      normal        := dataRegions.zip(regions).map { case (hit, region) => hit && region.normal.B }.reduce(_ || _)
      regionAllowed := permittedRegion
      state         := authorize
    }
  }

  physical.io.request.valid            := state === startAccess
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
  physical.io.response.ready           := state === waitAccess
  when(physical.io.response.fire) {
    assert(physical.io.response.bits.tag === command.tag, "Virtual LSU physical response tag mismatch")
    answer.data                   := physical.io.response.bits.data
    answer.misaligned             := physical.io.response.bits.misaligned
    answer.accessFault            := physical.io.response.bits.accessFault
    if (lineBits > 0) answer.line := physical.io.response.bits.line
    state                         := respond
  }

  val walking    = state === startWalk || state === waitWalk
  val accessing  = state === startAccess || state === waitAccess
  val pteFirst   = walker.io.access.bits.addr.pad(65)
  val pteRegions = matches(pteFirst, pteFirst + 7.U)

  val pteAllowed = pteRegions.zip(regions).map { case (hit, region) =>
    hit && (region.normal && region.readable && region.cacheable).B
  }.reduce(_ || _)

  // The Core owns a per-command PMP snapshot. Each permission decision is a
  // single-outstanding handshake; neither a stalled query nor its response
  // allows speculative cache or MMIO traffic. PTE accesses use S privilege.
  io.authorizationRequest.valid          := state === authorize ||
    (walking && walker.io.access.valid && !pteAuthorized && !pteAuthorizationPending && !pteError)
  io.authorizationRequest.bits.paddr     := Mux(walking, walker.io.access.bits.addr.pad(64), paddr)
  io.authorizationRequest.bits.size      := Mux(walking, 3.U, command.size)
  io.authorizationRequest.bits.read      := walking || readsData
  io.authorizationRequest.bits.write     := !walking && writesData
  io.authorizationRequest.bits.execute   := !walking && command.execute
  io.authorizationRequest.bits.isPte     := walking
  io.authorizationRequest.bits.privilege := Mux(walking, 1.U, command.privilege)
  when(io.authorizationRequest.fire) {
    when(walking) {
      pteAuthorizationPending := true.B
    }
      .otherwise(state := waitAuthorization)
  }
  io.authorizationResponse.ready         := pteAuthorizationPending || state === waitAuthorization
  when(io.authorizationResponse.fire) {
    when(pteAuthorizationPending) {
      pteAuthorizationPending := false.B
      pteAuthorized           := true.B
      pteError                := !io.authorizationResponse.bits || !pteAllowed
    }.otherwise {
      when(io.authorizationResponse.bits && regionAllowed)(state := startAccess)
        .otherwise { answer.accessFault := true.B; state := respond }
    }
  }
  io.cacheRequest.valid                  := Mux(
    walking,
    walker.io.access.valid && pteAuthorized && !pteError,
    accessing && physical.io.cacheRequest.valid
  )
  io.cacheRequest.bits                   := Mux(walking, walker.io.access.bits, physical.io.cacheRequest.bits)
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

  io.response.valid            := state === respond
  io.response.bits             := answer
  when(io.response.fire)(state := idle)
  io.observedTranslation.valid := walker.io.resp.fire
  io.observedTranslation.bits  := walker.io.resp.bits
  io.observedPhysical.valid    := physical.io.request.fire
  io.observedPhysical.bits     := physical.io.request.bits
  io.observedCache.valid       := io.cacheRequest.fire
  io.observedCache.bits.pte    := walking
  io.observedCache.bits.access := io.cacheRequest.bits
}
