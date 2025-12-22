package framework.memdomain.backend.banks

import chisel3._
import chisel3.util._

/**
 * Generic SRAM interface definitions
 */
class SramReadReq(val n: Int) extends Bundle {
  val addr    = UInt(log2Ceil(n).W)
  val fromDMA = Bool()
}

class SramReadResp(val w: Int) extends Bundle {
  val data    = UInt(w.W)
  val fromDMA = Bool()
}

class SramReadIO(val n: Int, val w: Int) extends Bundle {
  val req  = Flipped(Decoupled(new SramReadReq(n)))
  val resp = Decoupled(new SramReadResp(w))
}

class SramWriteReq(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val addr  = UInt(log2Ceil(n).W)
  val mask  = Vec(mask_len, Bool())
  val data  = UInt(w.W)
  val wmode = Bool() // true=accumulator mode, false=direct write mode
}

class SramWriteIO(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val req  = Flipped(Decoupled(new SramWriteReq(n, w, mask_len)))
  val resp = Decoupled(new SramWriteResp())
}

class SramWriteResp() extends Bundle {
  val ok = Bool()
}
