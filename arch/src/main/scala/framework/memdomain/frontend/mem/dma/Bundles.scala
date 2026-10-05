package framework.memdomain.frontend.mem.dma

import chisel3._
import freechips.rocketchip.rocket.MStatus

class BBReadRequest extends Bundle {
  val vaddr        = UInt(64.W)
  val len          = UInt(32.W)
  val status       = new MStatus
  val stride       = UInt(19.W)
  val groups       = UInt(6.W)
  val is_2d        = Bool()
  val pixel_bytes  = UInt(10.W)
  val source_width = UInt(10.W)
  val tile_width   = UInt(4.W)
}

class BBReadResponse(dataWidth: Int) extends Bundle {
  val data        = UInt(dataWidth.W)
  val last        = Bool()
  val addrcounter = UInt(16.W)
  val fault       = new DmaStatus
}

class BBWriteCommand extends Bundle {
  val vaddr = UInt(64.W)
  val len   = UInt(32.W)
}

class BBWriteData(dataWidth: Int) extends Bundle {
  val data = UInt(dataWidth.W)
  val last = Bool()
}

class BBWriteResponse extends Bundle {
  val done  = Bool()
  val fault = new DmaStatus
}
