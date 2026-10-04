package memcore.memory.cpu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import memcore.bus.chi.Params
import memcore.bus.chi.rnf.{CacheAccess, CacheAtomic, CacheResult}

case class CpuMemParams(chi: Params = Params(), tagBits: Int = 6) {
  require(chi.addressBits < 64 && tagBits > 0)
}

class CpuMemoryRequest(p: CpuMemParams) extends Bundle {
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
class CpuMemoryResponse(p: CpuMemParams, lineBits: Int = 0) extends Bundle {
  val tag         = UInt(p.tagBits.W)
  val data        = UInt(64.W)
  val misaligned  = Bool()
  val accessFault = Bool()
  val line        = UInt(lineBits.W)
}

class UncachedRequest(p: CpuMemParams) extends Bundle {
  val addr   = UInt(64.W)
  val tag    = UInt(p.tagBits.W)
  val size   = UInt(3.W)
  val write  = Bool()
  val data   = UInt(64.W)
  val atomic = UInt(4.W)
  val normal = Bool()
}

class UncachedResponse(p: CpuMemParams) extends Bundle {
  val tag   = UInt(p.tagBits.W)
  val data  = UInt(64.W)
  val error = Bool()
}

@instantiable
class CpuMem(p: CpuMemParams, lineBits: Int = 0) extends Module {

  @public val io = IO(new Bundle {
    val request          = Flipped(Decoupled(new CpuMemoryRequest(p)))
    val response         = Decoupled(new CpuMemoryResponse(p, lineBits))
    val cacheRequest     = Decoupled(new CacheAccess(p.chi))
    val cacheResponse    = Flipped(Decoupled(new CacheResult(lineBits)))
    val uncachedRequest  = Decoupled(new UncachedRequest(p))
    val uncachedResponse = Flipped(Decoupled(new UncachedResponse(p)))
  })

  val idle :: issueCache :: waitCache :: issueUncached :: waitUncached :: respond :: Nil = Enum(6)
  val state                                                                              = RegInit(idle)
  val command                                                                            = Reg(new CpuMemoryRequest(p))
  val answer                                                                             = Reg(new CpuMemoryResponse(p, lineBits))

  val mask = MuxLookup(command.size, 255.U(8.W))(Seq(
    0.U -> 1.U(8.W),
    1.U -> 3.U(8.W),
    2.U -> 15.U(8.W)
  ))

  io.request.ready := state === idle
  when(io.request.fire) {
    val r           = io.request.bits
    assert(r.size <= 3.U, "CPU memory size must be 1, 2, 4 or 8 bytes")
    assert(r.atomic <= CacheAtomic.SC.U, "CPU memory port does not accept Fence or unknown atomics")
    assert(r.atomic === CacheAtomic.None.U || (!r.write && r.size >= 2.U), "CPU atomic requires a 32 or 64 bit operand")
    val misaligned  = (r.addr & ((1.U(64.W) << r.size) - 1.U)).orR
    val accessFault = r.addr(63, p.chi.addressBits).orR || (!r.normal && r.atomic =/= CacheAtomic.None.U)
    command            := r
    answer.tag         := r.tag
    answer.data        := 0.U
    answer.misaligned  := misaligned
    answer.accessFault := !misaligned && accessFault
    state              := Mux(misaligned || accessFault, respond, Mux(r.cacheable, issueCache, issueUncached))
  }

  val atomic = command.atomic =/= CacheAtomic.None.U
  io.cacheRequest.valid               := state === issueCache
  io.cacheRequest.bits.addr           := Mux(atomic, command.addr, command.addr & ~7.U(64.W))(p.chi.addressBits - 1, 0)
  io.cacheRequest.bits.write          := command.write
  io.cacheRequest.bits.data           := Mux(atomic, command.data, command.data << (command.addr(2, 0) << 3))
  io.cacheRequest.bits.mask           := Mux(atomic, 255.U, Mux(command.write, mask << command.addr(2, 0), 0.U))
  io.cacheRequest.bits.atomic         := command.atomic
  io.cacheRequest.bits.atomicWord     := atomic && command.size === 2.U
  when(io.cacheRequest.fire)(state    := waitCache)
  io.cacheResponse.ready              := state === waitCache
  io.uncachedRequest.valid            := state === issueUncached
  io.uncachedRequest.bits.addr        := command.addr
  io.uncachedRequest.bits.tag         := command.tag
  io.uncachedRequest.bits.size        := command.size
  io.uncachedRequest.bits.write       := command.write
  io.uncachedRequest.bits.data        := command.data
  io.uncachedRequest.bits.atomic      := command.atomic
  io.uncachedRequest.bits.normal      := command.normal
  when(io.uncachedRequest.fire)(state := waitUncached)
  io.uncachedResponse.ready           := state === waitUncached

  val raw =
    Mux(state === waitUncached, io.uncachedResponse.bits.data, io.cacheResponse.bits.data >> (command.addr(2, 0) << 3))

  val load = MuxLookup(command.size, raw)(Seq(
    0.U -> Cat(Fill(56, command.signed && raw(7)), raw(7, 0)),
    1.U -> Cat(Fill(48, command.signed && raw(15)), raw(15, 0)),
    2.U -> Cat(Fill(32, command.signed && raw(31)), raw(31, 0))
  ))

  when(io.cacheResponse.fire || io.uncachedResponse.fire) {
    val error = Mux(state === waitUncached, io.uncachedResponse.bits.error, io.cacheResponse.bits.error)
    when(io.uncachedResponse.fire) {
      assert(io.uncachedResponse.bits.tag === command.tag, "Uncached response tag mismatch")
    }
    answer.data                   := Mux(
      error || command.write,
      0.U,
      Mux(atomic, Mux(state === waitUncached, io.uncachedResponse.bits.data, io.cacheResponse.bits.data), load)
    )
    answer.accessFault            := error
    if (lineBits > 0) answer.line := Mux(state === waitCache && !error, io.cacheResponse.bits.line, 0.U)
    state                         := respond
  }
  io.response.valid            := state === respond
  io.response.bits             := answer
  when(io.response.fire)(state := idle)
}
