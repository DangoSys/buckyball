package memcore.bus.chi.rnf

import chisel3._
import memcore.bus.chi.Params

class CacheAccess(p: Params) extends Bundle {
  val addr       = UInt(p.addressBits.W)
  val write      = Bool()
  val data       = UInt(64.W)
  val mask       = UInt(8.W)
  val atomic     = UInt(4.W)
  val atomicWord = Bool()
}

/**
 * Same-cycle hit access for a client that can fall back to `access` on a miss. The cache acts only
 * when `ready` (no demand operation, retirement or snoop in progress) and the line `hit`s with the
 * needed permission; a write then merges `data` under `mask` into the addressed doubleword.
 */
class CacheProbe(p: Params) extends Bundle {
  val valid = Input(Bool())
  val addr  = Input(UInt(p.addressBits.W))
  val write = Input(Bool())
  val data  = Input(UInt(64.W))
  val mask  = Input(UInt(8.W))
  val ready = Output(Bool())
  val hit   = Output(Bool())
  val value = Output(UInt(64.W))
}

object CacheAtomic {
  val None  = 0
  val Swap  = 1
  val Add   = 2
  val Xor   = 3
  val And   = 4
  val Or    = 5
  val Min   = 6
  val Max   = 7
  val MinU  = 8
  val MaxU  = 9
  val LR    = 10
  val SC    = 11
  val Fence = 12
}

/** `line` carries the whole 64-byte line for clients that consume full lines; it is absent by default. */
class CacheResult(lineBits: Int = 0) extends Bundle {
  val data  = UInt(64.W)
  val error = Bool()
  val line  = UInt(lineBits.W)
}

class CacheLineState(p: Params) extends Bundle {
  val valid    = Bool()
  val writable = Bool()
  val line     = UInt((p.addressBits - 6).W)
}
