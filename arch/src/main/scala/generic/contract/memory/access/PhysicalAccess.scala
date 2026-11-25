package framework.generic.contract.memory.access

import framework.generic.contract.memory.access.AccessParams
import framework.generic.contract.memory.access.UncachedRequest
import framework.generic.contract.memory.access.UncachedResponse

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.bus.chi.Params
import framework.generic.bus.chi.rnf.{CacheAccess, CacheAtomic, CacheResult}

case class AccessParams(chi: Params = Params(), tagBits: Int = 6) {
  require(chi.addressBits < 64 && tagBits > 0)
}

class UncachedRequest(p: AccessParams) extends Bundle {

  val addr  = UInt(64.W)
  val tag   = UInt(p.tagBits.W)
  val size  = UInt(3.W)
  val write = Bool()
  val data  = UInt(64.W)
}

class UncachedResponse(p: AccessParams) extends Bundle {

  val tag   = UInt(p.tagBits.W)
  val data  = UInt(64.W)
  val error = Bool()
}
