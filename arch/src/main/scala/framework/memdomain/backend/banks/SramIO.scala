package framework.memdomain.backend.banks

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig

/**
 * Generic SRAM interface definitions
 */
class SramReadReq(val b: GlobalConfig) extends Bundle {
  val addr    = UInt(log2Ceil(b.memDomain.bankEntries).W)
  val fromDMA = Bool()
}

class SramReadResp(val b: GlobalConfig) extends Bundle {
  val data    = UInt(b.memDomain.bankWidth.W)
  val fromDMA = Bool()
}

class SramReadIO(val b: GlobalConfig) extends Bundle {
  val req  = Flipped(Decoupled(new SramReadReq(b)))
  val resp = Decoupled(new SramReadResp(b))
}

class SramWriteReq(val b: GlobalConfig) extends Bundle {
  val addr  = UInt(log2Ceil(b.memDomain.bankEntries).W)
  val mask  = Vec(b.memDomain.bankMaskLen, Bool())
  val data  = UInt(b.memDomain.bankWidth.W)
  val wmode = Bool() // true=accumulator mode, false=direct write mode
}

class SramWriteIO(val b: GlobalConfig) extends Bundle {
  val req  = Flipped(Decoupled(new SramWriteReq(b)))
  val resp = Decoupled(new SramWriteResp(b))
}

class SramWriteResp(val b: GlobalConfig) extends Bundle {
  val ok      = Bool()
  val fromDMA = Bool()
}
