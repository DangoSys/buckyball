package memcore.memory.interlock

import chisel3._

class Dispatch(p: Params) extends Bundle { val id = UInt(p.idBits.W) }

class AccessInfo(p: Params) extends Bundle {
  val id        = UInt(p.idBits.W)
  val hasMemory = Bool()
  val base      = UInt(p.addressBits.W)
  val bytes     = UInt((p.addressBits + 1).W)
  val write     = Bool()
  val last      = Bool()
}

class CpuQuery(p: Params) extends Bundle {
  val valid                = Bool()
  val paddr                = UInt(p.addressBits.W)
  val sizeLog2             = UInt(3.W)
  val write                = Bool()
  val olderDispatchPending = Bool()
}

object MaintenanceOp { val Clean = 0; val CleanInvalidate = 1; val Invalidate = 2 }

class Maintenance(p: Params) extends Bundle {
  val tag       = UInt(p.idBits.W)
  val firstLine = UInt(p.addressBits.W)
  val lastLine  = UInt(p.addressBits.W)
  val op        = UInt(2.W)
}

class Tag(p: Params) extends Bundle { val tag = UInt(p.idBits.W) }

class Acknowledgement(p: Params) extends Bundle {
  val tag = UInt(p.idBits.W)
  val ok  = Bool()
}
