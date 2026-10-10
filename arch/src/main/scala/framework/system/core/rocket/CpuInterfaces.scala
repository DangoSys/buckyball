// See LICENSE.Berkeley and LICENSE.SiFive for the Rocket interface definitions.
package freechips.rocketchip.rocket

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.{CoreBundle, HasCoreParameters, TileInterrupts}

/** CPU-facing interfaces, independent of the upstream cache and TileLink implementations. */
class FrontendReq(implicit p: Parameters) extends CoreBundle()(p) {
  val pc          = UInt(vaddrBitsExtended.W)
  val speculative = Bool()
}

class FrontendExceptions extends Bundle {
  val pf = new Bundle { val inst = Bool() }
  val gf = new Bundle { val inst = Bool() }
  val ae = new Bundle { val inst = Bool() }
}

class FrontendResp(implicit p: Parameters) extends CoreBundle()(p) {
  val btb    = new BTBResp
  val pc     = UInt(vaddrBitsExtended.W)
  val data   = UInt((fetchWidth * coreInstBits).W)
  val mask   = Bits(fetchWidth.W)
  val xcpt   = new FrontendExceptions
  val replay = Bool()
}

class FrontendPerfEvents extends Bundle {
  val acquire = Bool()
  val tlbMiss = Bool()
}

class SFenceReq(implicit p: Parameters) extends CoreBundle()(p) {
  val rs1  = Bool()
  val rs2  = Bool()
  val addr = UInt(vaddrBits.W)
  val asid = UInt((asIdBits max 1).W)
  val hv   = Bool()
  val hg   = Bool()
}

class FrontendIO(implicit p: Parameters) extends CoreBundle()(p) {
  val might_request = Output(Bool())
  val clock_enabled = Input(Bool())
  val req           = Valid(new FrontendReq)
  val sfence        = Valid(new SFenceReq)
  val resp          = Flipped(Decoupled(new FrontendResp))
  val gpa           = Flipped(Valid(UInt(vaddrBitsExtended.W)))
  val gpa_is_pte    = Input(Bool())
  val btb_update    = Valid(new BTBUpdate)
  val bht_update    = Valid(new BHTUpdate)
  val ras_update    = Valid(new RASUpdate)
  val flush_icache  = Output(Bool())
  val npc           = Input(UInt(vaddrBitsExtended.W))
  val perf          = Input(new FrontendPerfEvents)
  val progress      = Output(Bool())
}

trait HasCoreMemOp extends HasCoreParameters {
  val addr = UInt(coreMaxAddrBits.W)

  val idx = Option.when(usingVM && lgCacheBlockBytes + log2Up(tileParams.dcache.get.nSets) > pgIdxBits)(
    UInt(coreMaxAddrBits.W)
  )

  val tag    = UInt((coreParams.dcacheReqTagBits + log2Ceil(dcacheArbPorts)).W)
  val cmd    = UInt(M_SZ.W)
  val size   = UInt(log2Ceil(log2Ceil(coreDataBytes) + 1).W)
  val signed = Bool()
  val dprv   = UInt(PRV.SZ.W)
  val dv     = Bool()
}

trait HasCoreData extends HasCoreParameters {
  val data = UInt(coreDataBits.W)
  val mask = UInt(coreDataBytes.W)
}

class HellaCacheReqInternal(implicit p: Parameters) extends CoreBundle()(p) with HasCoreMemOp {
  val phys     = Bool()
  val no_resp  = Bool()
  val no_alloc = Bool()
  val no_xcpt  = Bool()
}

class HellaCacheReq(implicit p: Parameters) extends HellaCacheReqInternal()(p) with HasCoreData

class HellaCacheResp(implicit p: Parameters) extends CoreBundle()(p) with HasCoreMemOp with HasCoreData {
  val replay           = Bool()
  val has_data         = Bool()
  val data_word_bypass = UInt(coreDataBits.W)
  val data_raw         = UInt(coreDataBits.W)
  val store_data       = UInt(coreDataBits.W)
}

class AlignmentExceptions extends Bundle {
  val ld = Bool()
  val st = Bool()
}

class HellaCacheExceptions extends Bundle {
  val ma = new AlignmentExceptions
  val pf = new AlignmentExceptions
  val gf = new AlignmentExceptions
  val ae = new AlignmentExceptions
}

class HellaCacheWriteData(implicit p: Parameters) extends CoreBundle()(p) with HasCoreData

class HellaCachePerfEvents extends Bundle {
  val acquire                    = Bool()
  val release                    = Bool()
  val grant                      = Bool()
  val tlbMiss                    = Bool()
  val blocked                    = Bool()
  val canAcceptStoreThenLoad     = Bool()
  val canAcceptStoreThenRMW      = Bool()
  val canAcceptLoadThenLoad      = Bool()
  val storeBufferEmptyAfterLoad  = Bool()
  val storeBufferEmptyAfterStore = Bool()
}

class HellaCacheIO(implicit p: Parameters) extends CoreBundle()(p) {
  val req                = Decoupled(new HellaCacheReq)
  val s1_kill            = Output(Bool())
  val s1_data            = Output(new HellaCacheWriteData)
  val s2_nack            = Input(Bool())
  val s2_nack_cause_raw  = Input(Bool())
  val s2_kill            = Output(Bool())
  val s2_uncached        = Input(Bool())
  val s2_paddr           = Input(UInt(paddrBits.W))
  val resp               = Flipped(Valid(new HellaCacheResp))
  val replay_next        = Input(Bool())
  val s2_xcpt            = Input(new HellaCacheExceptions)
  val s2_gpa             = Input(UInt(vaddrBitsExtended.W))
  val s2_gpa_is_pte      = Input(Bool())
  val uncached_resp      = Option.when(tileParams.dcache.get.separateUncachedResp)(Flipped(Decoupled(new HellaCacheResp)))
  val ordered            = Input(Bool())
  val store_pending      = Input(Bool())
  val perf               = Input(new HellaCachePerfEvents)
  val keep_clock_enabled = Output(Bool())
  val clock_enabled      = Input(Bool())
}

class PTWPerfEvents extends Bundle {
  val l2miss   = Bool()
  val l2hit    = Bool()
  val pte_miss = Bool()
  val pte_hit  = Bool()
}

class DatapathPTWIO(implicit p: Parameters) extends CoreBundle()(p) {
  val ptbr          = Input(new PTBR)
  val hgatp         = Input(new PTBR)
  val vsatp         = Input(new PTBR)
  val sfence        = Flipped(Valid(new SFenceReq))
  val status        = Input(new MStatus)
  val hstatus       = Input(new HStatus)
  val gstatus       = Input(new MStatus)
  val pmp           = Input(Vec(nPMPs, new PMP))
  val perf          = Output(new PTWPerfEvents)
  val customCSRs    = Flipped(coreParams.customCSRs)
  val clock_enabled = Output(Bool())
}

class CoreInterrupts(val hasBeu: Boolean)(implicit p: Parameters) extends TileInterrupts()(p) {
  val buserror = Option.when(hasBeu)(Bool())
}
