package hier.tile.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.bus.chi.Opcode
import memcore.bus.chi.rnf.CacheAccess
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.cpu.{
  CpuMemParams,
  CpuMemoryRequest,
  PhysicalAuthorization,
  PhysicalRegion,
  UncachedRequest,
  UncachedResponse,
  VirtualMemory,
  VirtualMemoryRequest,
  VirtualMemoryResponse
}
import memcore.memory.coherence.configs.CoherenceParams

/** Independent verification composition: one virtual client and the two-RN-F shared-L2 cache system. */
@instantiable
class VirtualCacheSystem(p: CoherenceParams, regions: Seq[PhysicalRegion]) extends Module {
  private val cp = CpuMemParams(p.chi, tagBits = 6)

  @public
  val io = IO(new Bundle {
    val active                = Input(Bool())
    val request               = Flipped(Decoupled(new VirtualMemoryRequest(cp)))
    val response              = Decoupled(new VirtualMemoryResponse(cp))
    val authorizationRequest  = Decoupled(new PhysicalAuthorization)
    val authorizationResponse = Flipped(Decoupled(Bool()))
    val memoryReq             = Decoupled(new LineRequest(p.chi))
    val memoryResp            = Flipped(Decoupled(new LineResponse(p.chi)))
    val uncachedRequest       = Decoupled(new UncachedRequest(cp))
    val uncachedResponse      = Flipped(Decoupled(new UncachedResponse(cp)))
    val observedTranslation   = Output(Valid(new memcore.memory.mmu.Response(p.chi)))
    val observedPhysical      = Output(Valid(new CpuMemoryRequest(cp)))

    val observedCache = Output(Valid(new Bundle {
      val pte    = Bool()
      val access = new CacheAccess(p.chi)
    }))

    val observedEviction = Output(Valid(UInt(p.chi.addressBits.W)))
    val outstanding      = Output(UInt(log2Ceil(p.mshrEntries + 1).W))
  })

  val virtualMemory = Instantiate(new VirtualMemory(cp, regions))
  val caches        = Instantiate(new CacheSystem(p))
  virtualMemory.io.active     := io.active
  virtualMemory.io.request <> io.request
  io.response <> virtualMemory.io.response
  io.authorizationRequest <> virtualMemory.io.authorizationRequest
  virtualMemory.io.authorizationResponse <> io.authorizationResponse
  io.uncachedRequest <> virtualMemory.io.uncachedRequest
  virtualMemory.io.uncachedResponse <> io.uncachedResponse
  caches.io.active            := io.active
  caches.io.blockRequesterRsp := false.B
  caches.io.access(0) <> virtualMemory.io.cacheRequest
  virtualMemory.io.cacheResponse <> caches.io.result(0)
  caches.io.access(1).valid   := false.B
  caches.io.access(1).bits    := 0.U.asTypeOf(caches.io.access(1).bits)
  caches.io.result(1).ready   := false.B
  when(caches.io.result(1).valid)(assert(false.B, "Virtual LSU inactive requester returned data"))
  io.memoryReq <> caches.io.memoryReq
  caches.io.memoryResp <> io.memoryResp
  io.outstanding              := caches.io.outstanding
  io.observedTranslation      := virtualMemory.io.observedTranslation
  io.observedPhysical         := virtualMemory.io.observedPhysical
  io.observedCache            := virtualMemory.io.observedCache
  io.observedEviction.valid   := caches.io.observedReq.valid && caches.io.observedReq.bits.opcode === Opcode.Evict.U
  io.observedEviction.bits    := caches.io.observedReq.bits.addr
}
