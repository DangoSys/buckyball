package memcore.bus.axi4

import chisel3._
import chisel3.util._

class Address(p: Params) extends Bundle {
  val id     = UInt(p.idBits.W)
  val addr   = UInt(p.addressBits.W)
  val len    = UInt(8.W)
  val size   = UInt(3.W)
  val burst  = UInt(2.W)
  val lock   = Bool()
  val cache  = UInt(4.W)
  val prot   = UInt(3.W)
  val qos    = UInt(4.W)
  val region = UInt(4.W)
}

class WriteData(p: Params) extends Bundle {
  val data = UInt(p.dataBits.W)
  val strb = UInt(p.bytes.W)
  val last = Bool()
}

class ReadData(p: Params) extends Bundle {
  val id   = UInt(p.idBits.W)
  val data = UInt(p.dataBits.W)
  val resp = UInt(2.W)
  val last = Bool()
}

class WriteResponse(p: Params) extends Bundle {
  val id   = UInt(p.idBits.W)
  val resp = UInt(2.W)
}

/** AXI4 memory-mapped master. Optional user sidebands are absent. */
class Port(p: Params) extends Bundle {
  val aw = Decoupled(new Address(p))
  val w  = Decoupled(new WriteData(p))
  val b  = Flipped(Decoupled(new WriteResponse(p)))
  val ar = Decoupled(new Address(p))
  val r  = Flipped(Decoupled(new ReadData(p)))
}
