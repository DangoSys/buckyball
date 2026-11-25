package framework.hart.memory.access

import framework.generic.contract.memory.access.UncachedRequest
import framework.generic.contract.memory.access.UncachedResponse

import framework.hart.memory.access.configs.{Params => ModuleParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.bus.chi.Params
import framework.generic.bus.chi.rnf.{CacheAccess, CacheAtomic, CacheResult}

class AccessRequest(b: ModuleParam) extends Bundle {
  val p = b.memory

  val addr      = UInt(64.W)
  val tag       = UInt(p.tagBits.W)
  val size      = UInt(3.W)
  val write     = Bool()
  val signed    = Bool()
  val data      = UInt(64.W)
  val atomic    = UInt(4.W)
  val cacheable = Bool()
  val normal    = Bool()
}

/** `line` is the whole cached line for full-line clients; it is absent by default. */
class AccessResponse(b: ModuleParam) extends Bundle {
  val p        = b.memory
  val lineBits = b.lineBits

  val tag         = UInt(p.tagBits.W)
  val data        = UInt(64.W)
  val misaligned  = Bool()
  val accessFault = Bool()
  val line        = UInt(lineBits.W)
}

@instantiable
class PhysicalAccess(b: ModuleParam) extends Module {
  val p        = b.memory
  val lineBits = b.lineBits

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new AccessRequest(b)))
    val response = Decoupled(new AccessResponse(b))

    val cacheRequest = Decoupled(new CacheAccess(p.chi))

    val cacheResponse    = Flipped(Decoupled(new CacheResult(lineBits)))
    val uncachedRequest  = Decoupled(new UncachedRequest(b.memory))
    val uncachedResponse = Flipped(Decoupled(new UncachedResponse(b.memory)))
  })

  // A legal access goes to the cache or uncached port in the cycle it is accepted and its result
  // returns in the cycle the port answers; only a misaligned or faulting access is answered here.
  val idle :: waitCache :: waitUncached :: respond :: Nil = Enum(4)
  val state                                               = RegInit(idle)
  val command                                             = Reg(new AccessRequest(b))
  val answer                                              = Reg(new AccessResponse(b))

  val r           = io.request.bits
  val misaligned  = (r.addr & ((1.U(64.W) << r.size) - 1.U)).orR
  val accessFault = r.addr(63, p.chi.addressBits).orR || (r.normal && !r.cacheable) ||
    (!r.normal && (r.cacheable || r.atomic =/= CacheAtomic.None.U))
  val faulting    = misaligned || accessFault

  val requestMask = MuxLookup(r.size, 255.U(8.W))(Seq(
    0.U -> 1.U(8.W),
    1.U -> 3.U(8.W),
    2.U -> 15.U(8.W)
  ))

  val requestAtomic = r.atomic =/= CacheAtomic.None.U

  io.cacheRequest.valid           := state === idle && io.request.valid && !faulting && r.cacheable
  io.cacheRequest.bits.addr       := Mux(requestAtomic, r.addr, r.addr & ~7.U(64.W))(p.chi.addressBits - 1, 0)
  io.cacheRequest.bits.write      := r.write
  io.cacheRequest.bits.data       := Mux(requestAtomic, r.data, r.data << (r.addr(2, 0) << 3))
  io.cacheRequest.bits.mask       := Mux(requestAtomic, 255.U, Mux(r.write, requestMask << r.addr(2, 0), 0.U))
  io.cacheRequest.bits.atomic     := r.atomic
  io.cacheRequest.bits.atomicWord := requestAtomic && r.size === 2.U
  io.uncachedRequest.valid        := state === idle && io.request.valid && !faulting && !r.cacheable
  io.uncachedRequest.bits.addr    := r.addr
  io.uncachedRequest.bits.tag     := r.tag
  io.uncachedRequest.bits.size    := r.size
  io.uncachedRequest.bits.write   := r.write
  io.uncachedRequest.bits.data    := r.data
  io.request.ready                := state === idle &&
    (faulting || Mux(r.cacheable, io.cacheRequest.ready, io.uncachedRequest.ready))

  when(io.request.fire) {
    assert(r.size <= 3.U, "Memory access size must be 1, 2, 4 or 8 bytes")
    assert(r.atomic <= CacheAtomic.SC.U, "Memory access port does not accept Fence or unknown atomics")
    assert(
      r.atomic === CacheAtomic.None.U || (!r.write && r.size >= 2.U),
      "Memory atomic requires a 32 or 64 bit operand"
    )
    command                       := r
    answer.tag                    := r.tag
    answer.data                   := 0.U
    if (lineBits > 0) answer.line := 0.U
    answer.misaligned             := misaligned
    answer.accessFault            := !misaligned && accessFault
    state                         := Mux(faulting, respond, Mux(r.cacheable, waitCache, waitUncached))
  }

  val atomic = command.atomic =/= CacheAtomic.None.U
  val raw    =
    Mux(state === waitUncached, io.uncachedResponse.bits.data, io.cacheResponse.bits.data >> (command.addr(2, 0) << 3))

  val load = MuxLookup(command.size, raw)(Seq(
    0.U -> Cat(Fill(56, command.signed && raw(7)), raw(7, 0)),
    1.U -> Cat(Fill(48, command.signed && raw(15)), raw(15, 0)),
    2.U -> Cat(Fill(32, command.signed && raw(31)), raw(31, 0))
  ))

  val error = Mux(state === waitUncached, io.uncachedResponse.bits.error, io.cacheResponse.bits.error)
  io.cacheResponse.ready       := state === waitCache && io.response.ready
  io.uncachedResponse.ready    := state === waitUncached && io.response.ready
  io.response.valid            := state === respond ||
    (state === waitCache && io.cacheResponse.valid) || (state === waitUncached && io.uncachedResponse.valid)
  io.response.bits             := answer
  when(state =/= respond) {
    io.response.bits.data                   := Mux(error || command.write, 0.U, Mux(atomic, io.cacheResponse.bits.data, load))
    io.response.bits.accessFault            := error
    if (lineBits > 0) io.response.bits.line := Mux(state === waitCache && !error, io.cacheResponse.bits.line, 0.U)
  }
  when(io.uncachedResponse.fire) {
    assert(io.uncachedResponse.bits.tag === command.tag, "Uncached response tag mismatch")
  }
  when(io.response.fire)(state := idle)
}
