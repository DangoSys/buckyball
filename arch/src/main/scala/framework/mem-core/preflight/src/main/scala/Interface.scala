package memcore.memory.preflight

import chisel3._

/** Shape describes transport bytes, including any read beyond the user's useful payload. */
class Command(p: Params) extends Bundle {
  val id           = UInt(p.idBits.W)
  val baseVA       = UInt(64.W)
  val rows         = UInt(32.W)
  val columns      = UInt(32.W)
  val spanBytes    = UInt(32.W)
  val columnStride = UInt(32.W)
  val rowStride    = UInt(32.W)
  val write        = Bool()
  val mode         = UInt(4.W)
  val rootPpn      = UInt(44.W)
  val privilege    = UInt(2.W)
  val sum          = Bool()
  val mxr          = Bool()
}

object Error {
  val Ok          = 0; val Shape    = 1; val Overflow = 2; val PageFault = 3
  val AccessFault = 4; val Capacity = 5; val Context  = 6
}

/** Failure returns one terminal zero-PA record. The caller retains successful maps and IDs until DMA retirement. */
class PreparedSegment(p: Params) extends Bundle {
  val id    = UInt(p.idBits.W)
  val va    = UInt(64.W)
  val pa    = UInt(p.bus.addressBits.W)
  val bytes = UInt(13.W)
  val write = Bool()
  val last  = Bool()
  val error = UInt(3.W)
}

class Authorization(p: Params) extends Bundle {
  val id        = UInt(p.idBits.W)
  val pa        = UInt(p.bus.addressBits.W)
  val bytes     = UInt(13.W)
  val write     = Bool()
  val isPte     = Bool()
  val privilege = UInt(2.W)
}

class Permission(p: Params) extends Bundle {
  val id    = UInt(p.idBits.W)
  val allow = Bool()
}
