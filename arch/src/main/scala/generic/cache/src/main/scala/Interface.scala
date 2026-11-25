package framework.generic.memory.cache

import chisel3._
import chisel3.util._
import framework.generic.memory.cache.configs.CacheParams

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

class CacheLine(p: CacheParams) extends Bundle {
  val address = Input(UInt(p.addressBits.W))
  val hit     = Output(Bool())
  val way     = Output(UInt(p.wayBits.W))
  val dirty   = Output(Bool())
}

/**
 * Whether any line of the 4 KiB page of `address` may be resident: the lines counted by page
 * bucket, so a page that shares its bucket with a resident one answers true.
 */
class CachePage(p: CacheParams) extends Bundle {
  val address  = Input(UInt(p.addressBits.W))
  val resident = Output(Bool())
}

class CacheIO(p: CacheParams) extends Bundle {
  val request  = Flipped(Decoupled(new CacheRequest(p)))
  val response = Decoupled(new CacheResponse(p))
}
