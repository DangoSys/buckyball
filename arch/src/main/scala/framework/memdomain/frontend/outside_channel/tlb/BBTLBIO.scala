package framework.memdomain.frontend.outside_channel.tlb

import chisel3._
import chisel3.util.{Decoupled, Valid}
import chisel3.util.log2Ceil
import framework.top.GlobalConfig
import freechips.rocketchip.rocket.{HStatus, MStatus}
import freechips.rocketchip.rocket.constants.MemoryOpConstants

// TLB Exception types
class TLBExceptions extends Bundle {
  val ld   = Bool()
  val st   = Bool()
  val inst = Bool()
}

// TLB Request
class BBTLBReq(val lgMaxSize: Int, val vaddrBits: Int, val xLen: Int) extends Bundle {
  val vaddr       = UInt(vaddrBits.W)
  val passthrough = Bool()
  val size        = UInt(log2Ceil(lgMaxSize + 1).W)
  val cmd         = Bits(5.W) // M_SZ = 5
  val prv         = UInt(2.W) // PRV.SZ = 2
  val v           = Bool()
  val status      = new MStatus
}

// TLB Response
class BBTLBResp(val lgMaxSize: Int, val paddrBits: Int, val vaddrBits: Int) extends Bundle {
  val miss         = Bool()
  val paddr        = UInt(paddrBits.W)
  val gpa          = UInt(vaddrBits.W)
  val gpa_is_pte   = Bool()
  val pf           = new TLBExceptions
  val gf           = new TLBExceptions
  val ae           = new TLBExceptions
  val ma           = new TLBExceptions
  val cacheable    = Bool()
  val must_alloc   = Bool()
  val prefetchable = Bool()
  val size         = UInt(log2Ceil(lgMaxSize + 1).W)
  val cmd          = UInt(5.W) // M_SZ = 5
}

// TLB Exception IO
class BBTLBExceptionIO extends Bundle {
  val interrupt   = Output(Bool())
  val flush_retry = Input(Bool())
  val flush_skip  = Input(Bool())

  def flush(dummy: Int = 0): Bool = flush_retry || flush_skip
}

// Page Table Base Register
class BBTLBPTBR(val paddrBits: Int, val pgIdxBits: Int, val xLen: Int) extends Bundle {
  val modeBits    = if (xLen == 32) 1 else 4
  val maxASIdBits = if (xLen == 32) 9 else 16
  val mode        = UInt(modeBits.W)
  val asid        = UInt(maxASIdBits.W)
  val ppn         = UInt((paddrBits - pgIdxBits).W)
}

// PTW Request
class BBTLBPTWReq(val vaddrBits: Int, val pgIdxBits: Int) extends Bundle {
  val vpnBits  = vaddrBits - pgIdxBits
  val addr     = UInt(vpnBits.W)
  val need_gpa = Bool()
  val vstage1  = Bool()
  val stage2   = Bool()
}

// PTE (Page Table Entry) - Simplified
class BBTLBPTE(val paddrBits: Int, val pgIdxBits: Int) extends Bundle {
  val ppnBits               = paddrBits - pgIdxBits
  val ppn                   = UInt(ppnBits.W)
  val reserved_for_future   = UInt(10.W)
  val reserved_for_software = Bits(2.W)
  val d                     = Bool() // dirty
  val a                     = Bool() // access
  val g                     = Bool() // global
  val u                     = Bool() // user
  val x                     = Bool() // executable
  val w                     = Bool() // writable
  val r                     = Bool() // readable
  val v                     = Bool() // valid

  def sr(): Bool = v && r
  def sw(): Bool = v && w && d
  def sx(): Bool = v && x
}

// PTW Response
class BBTLBPTWResp(
  val vaddrBits: Int,
  val paddrBits: Int,
  val pgIdxBits: Int,
  val pgLevels:  Int)
    extends Bundle {
  val ae_ptw               = Bool()
  val ae_final             = Bool()
  val pf                   = Bool()
  val gf                   = Bool()
  val hr                   = Bool()
  val hw                   = Bool()
  val hx                   = Bool()
  val pte                  = new BBTLBPTE(paddrBits, pgIdxBits)
  val level                = UInt(log2Ceil(pgLevels).W)
  val fragmented_superpage = Bool()
  val homogeneous          = Bool()
  val gpa                  = Valid(UInt(vaddrBits.W))
  val gpa_is_pte           = Bool()
}

// Simplified CustomCSR IO wrapper - without Parameters dependency
class BBCustomCSRIO(val xLen: Int) extends Bundle {
  val ren   = Output(Bool())
  val wen   = Output(Bool())
  val wdata = Output(UInt(xLen.W))
  val value = Output(UInt(xLen.W))
  val stall = Input(Bool())
  val set   = Input(Bool())
  val sdata = Input(UInt(xLen.W))
}

// Simplified CustomCSRs Bundle - matches rocket-chip interface without Parameters
class BBCustomCSRs(val xLen: Int) extends Bundle {
  // Empty by default - no custom CSRs defined
  val csrs = Vec(0, new BBCustomCSRIO(xLen))
}

// Simplified PMP - without Parameters dependency
class BBPMP(val paddrBits: Int) extends Bundle {

  val cfg = new Bundle {
    val l   = Bool()
    val res = UInt(2.W) // Reserved field
    val a   = UInt(2.W)
    val x   = Bool()
    val w   = Bool()
    val r   = Bool()
  }

  val addr = UInt((paddrBits - 2).W)
  val mask = UInt(paddrBits.W)
}

// PTW IO
class BBTLBPTWIO(val b: GlobalConfig) extends Bundle {
  val vaddrBits = b.core.vaddrBits
  val paddrBits = b.core.paddrBits
  val pgIdxBits = b.core.pgIdxBits
  val xLen      = b.core.xLen
  val pgLevels  = if (xLen == 32) 2 else 4 // Simplified: assume SV39 for 64-bit
  val nPMPs     = b.core.nPMPs

  val req        = Decoupled(Valid(new BBTLBPTWReq(vaddrBits, pgIdxBits)))
  val resp       = Flipped(Valid(new BBTLBPTWResp(vaddrBits, paddrBits, pgIdxBits, pgLevels)))
  val ptbr       = Input(new BBTLBPTBR(paddrBits, pgIdxBits, xLen))
  val hgatp      = Input(new BBTLBPTBR(paddrBits, pgIdxBits, xLen))
  val vsatp      = Input(new BBTLBPTBR(paddrBits, pgIdxBits, xLen))
  val status     = Input(new MStatus)
  val hstatus    = Input(new HStatus)
  val gstatus    = Input(new MStatus)
  val pmp        = Input(Vec(nPMPs, new BBPMP(paddrBits)))
  val customCSRs = Flipped(new BBCustomCSRs(xLen))
}

// TLB Client IO (used in TLBCluster)
class BBTLBIO(val b: GlobalConfig) extends Bundle {
  val lgMaxSize = log2Ceil(b.core.coreDataBytes)
  val req       = Flipped(Valid(new BBTLBReq(lgMaxSize, b.core.vaddrBits, b.core.xLen)))
  val resp      = Valid(new BBTLBResp(lgMaxSize, b.core.paddrBits, b.core.vaddrBits))
}
