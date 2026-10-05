package memcore.memory.uncached_ram

import chisel3._

class Request(p: Params) extends Bundle {
  val addr   = UInt(64.W)
  val tag    = UInt(p.tagBits.W)
  val size   = UInt(3.W)
  val write  = Bool()
  val data   = UInt(64.W)
  val atomic = UInt(4.W)
}

class Response(p: Params) extends Bundle {
  val tag   = UInt(p.tagBits.W)
  // Ordinary loads return low transfer bits; LR.W/AMO.W return sign-extended old values.
  val data  = UInt(64.W)
  val error = Bool()
}

/** Physical control range; DMA data never enters the CPU line transport. */
class ExternalAccess(p: Params) extends Bundle {
  val valid = Bool()
  val addr  = UInt(p.line.addressBits.W)
  val bytes = UInt(13.W)
  val write = Bool()
}
