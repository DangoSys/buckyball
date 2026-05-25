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
  val addr        = UInt(log2Ceil(b.memDomain.mmioBankEntries).W)
  val byte_offset = UInt(log2Ceil(b.memDomain.mmioBankWidth / 8).W)
}

class MmioBankReadResp(b: GlobalConfig) extends Bundle {
  val data = UInt(b.memDomain.mmioReadWidth.W)
}

class MmioBankReadIO(b: GlobalConfig) extends Bundle {
  val req  = Flipped(Decoupled(new MmioBankReadReq(b)))
  val resp = Decoupled(new MmioBankReadResp(b))
}

// ===== Ball-facing read port (carried over BlinkIO) =====
class MmioReadReq(b: GlobalConfig) extends Bundle {
  // Byte offset relative to the region base.
  // Sized to cover the largest possible region (= totalBytes).
  val rel_addr = UInt(log2Ceil(b.memDomain.mmioTotalBytes).W)
}

class MmioReadResp(b: GlobalConfig) extends Bundle {
  val data = UInt(b.memDomain.mmioReadWidth.W)
}

// ===== Region table alloc/dealloc ports =====
class MmioAllocReq(b: GlobalConfig) extends Bundle {
  val main_bank = UInt(log2Up(b.memDomain.bankNum).W)
  val mmio_addr = UInt(log2Ceil(b.memDomain.mmioTotalBytes).W)
  val size_rows = UInt(8.W) // 0 means dealloc
}

class MmioRegionEntry(b: GlobalConfig) extends Bundle {
  val valid     = Bool()
  val mmio_addr = UInt(log2Ceil(b.memDomain.mmioTotalBytes).W)
  val size_rows = UInt(8.W)
}
