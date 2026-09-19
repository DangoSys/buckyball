package memcore.memory.cache

import chisel3._
import memcore.bus.chi.Params

class CacheAccess(p: Params) extends Bundle {
  val addr   = UInt(p.addressBits.W)
  val write  = Bool()
  val data   = UInt(64.W)
  val mask   = UInt(8.W)
  val atomic = UInt(4.W)
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

class CacheResult extends Bundle {
  val data  = UInt(64.W)
  val error = Bool()
}

class CacheLineState(p: Params) extends Bundle {
  val valid    = Bool()
  val writable = Bool()
  val line     = UInt((p.addressBits - 6).W)
}
