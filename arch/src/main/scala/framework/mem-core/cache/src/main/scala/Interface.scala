package memcore.memory.cache

import chisel3._
import chisel3.util._
import memcore.memory.cache.configs.CacheParams

object CacheOp {
  val Lookup     = 0
  val Read       = 1
  val Write      = 2
  val Fill       = 3
  val Invalidate = 4
}

class CacheRequest(p: CacheParams) extends Bundle {
  val id       = UInt(p.idBits.W)
  val op       = UInt(3.W)
  // Read and Invalidate address a physical set/way; all operations are line aligned.
  val addr     = UInt(p.addressBits.W)
  val way      = UInt(p.wayBits.W)
  val data     = UInt(p.lineBits.W)
  val mask     = UInt(p.lineBytes.W)
  val metadata = UInt(p.metadataBits.W)
  // Eligibility constrains miss replacement candidates, not tag hits.
  val eligible = UInt(p.ways.W)
}

class CacheResponse(p: CacheParams) extends Bundle {
  val id         = UInt(p.idBits.W)
  val hit        = Bool()
  val available  = Bool()
  val way        = UInt(p.wayBits.W)
  val entryValid = Bool()
  val addr       = UInt(p.addressBits.W)
  val data       = UInt(p.lineBits.W)
  val metadata   = UInt(p.metadataBits.W)
}

class CacheIO(p: CacheParams) extends Bundle {
  val request  = Flipped(Decoupled(new CacheRequest(p)))
  val response = Decoupled(new CacheResponse(p))
}
