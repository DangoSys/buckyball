package framework.memdomain.backend.shared

import chisel3._
import chisel3.util._

/** Tile shared SRAM byte space; not a Core virtual-bank or private-bank address. */
class SharedPhysicalRequest(dataBits: Int) extends Bundle {
  val address = UInt(64.W)
  val write   = Bool()
  val data    = UInt(dataBits.W)
  val mask    = UInt((dataBits / 8).W)
}

class SharedPhysicalResponse(dataBits: Int) extends Bundle {
  val data  = UInt(dataBits.W)
  val error = Bool()
}

/** One outstanding transaction per port; a write response acknowledges actual SRAM completion. */
class SharedPhysicalPort(dataBits: Int) extends Bundle {
  val request  = Decoupled(new SharedPhysicalRequest(dataBits))
  val response = Flipped(Decoupled(new SharedPhysicalResponse(dataBits)))
}
