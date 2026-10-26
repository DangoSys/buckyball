package framework.ip.memory.bank

import chisel3._
import chisel3.util._
import framework.acc.memdomain.configs.MemDomainParam

/**
 * Generic SRAM interface definitions
 */
class SramReadReq(val b: MemDomainParam) extends Bundle {
  val addr = UInt(16.W)
}

class SramReadResp(val b: MemDomainParam) extends Bundle {
  val data = UInt(b.bankWidth.W)
}

class SramReadIO(val b: MemDomainParam) extends Bundle {
  val req  = Flipped(Decoupled(new SramReadReq(b)))
  val resp = Decoupled(new SramReadResp(b))
}

class SramWriteReq(val b: MemDomainParam) extends Bundle {
  val addr = UInt(16.W)
  val mask = Vec(b.bankMaskLen, Bool())
  val data = UInt(b.bankWidth.W)
}

class SramWriteIO(val b: MemDomainParam) extends Bundle {
  val req  = Flipped(Decoupled(new SramWriteReq(b)))
  val resp = Decoupled(new SramWriteResp(b))
}

class SramWriteResp(val b: MemDomainParam) extends Bundle {
  val ok = Bool()
}
