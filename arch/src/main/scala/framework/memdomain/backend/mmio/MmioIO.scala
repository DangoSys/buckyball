package framework.memdomain.backend.mmio

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig

/**
 * Internal bundles for the MMIO subsystem.
 *
 * Naming convention:
 *   *Req / *Resp = Bundles that travel through Decoupled channels
 *   *Port        = Bundles that group req+resp together
 */

// ===== Per-MmioBank internal read port =====
class MmioBankReadReq(b: GlobalConfig) extends Bundle {
  val addr = UInt(log2Ceil(b.memDomain.mmioBankEntries).W)
}

class MmioBankReadResp(b: GlobalConfig) extends Bundle {
  val data = UInt(b.memDomain.mmioReadWidth.W)
}

class MmioBankReadIO(b: GlobalConfig) extends Bundle {
  val req  = Flipped(Decoupled(new MmioBankReadReq(b)))
  val resp = Decoupled(new MmioBankReadResp(b))
}

class MmioBankWriteReq(b: GlobalConfig) extends Bundle {
  val addr = UInt(log2Ceil(b.memDomain.mmioBankEntries).W)
  val data = UInt(b.memDomain.mmioReadWidth.W)
}

class MmioBankWriteIO(b: GlobalConfig) extends Bundle {
  val req = Flipped(Decoupled(new MmioBankWriteReq(b)))
}

// ===== Ball-facing MMIO ports (carried over BlinkIO) =====
class MmioReadReq(b: GlobalConfig) extends Bundle {
  // MMIO is one globally encoded byte address space.
  val addr = UInt(log2Ceil(b.memDomain.mmioTotalBytes).W)
}

class MmioReadResp(b: GlobalConfig) extends Bundle {
  val data = UInt(b.memDomain.mmioReadWidth.W)
}

class MmioWriteReq(b: GlobalConfig) extends Bundle {
  val addr = UInt(log2Ceil(b.memDomain.mmioTotalBytes).W)
  val data = UInt(b.memDomain.mmioReadWidth.W)
}
