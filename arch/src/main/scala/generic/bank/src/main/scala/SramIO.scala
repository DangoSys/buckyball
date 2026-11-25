package framework.generic.memory.bank

import chisel3._
import chisel3.util._

/** Row read and write ports of an SRAM of `width` bits a row. */
class SramReadReq extends Bundle {
  val addr = UInt(16.W)
}

class SramReadResp(width: Int) extends Bundle {
  val data = UInt(width.W)
}

class SramReadIO(width: Int) extends Bundle {
  val req  = Flipped(Decoupled(new SramReadReq))
  val resp = Decoupled(new SramReadResp(width))
}

class SramWriteReq(width: Int, maskLen: Int) extends Bundle {
  val addr = UInt(16.W)
  val mask = Vec(maskLen, Bool())
  val data = UInt(width.W)
}

class SramWriteIO(width: Int, maskLen: Int) extends Bundle {
  val req  = Flipped(Decoupled(new SramWriteReq(width, maskLen)))
  val resp = Decoupled(new SramWriteResp)
}

class SramWriteResp extends Bundle {
  val ok = Bool()
}
