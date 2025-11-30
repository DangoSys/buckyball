package framework.memdomain.tlb

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile.{CoreBundle, CoreModule}
import freechips.rocketchip.tilelink.TLEdgeOut

import framework.builtin.util.Util._

class BBTLBReq(val lgMaxSize: Int)(implicit p: Parameters) extends CoreBundle {
  val tlb_req = new TLBReq(lgMaxSize)
  val status = new MStatus
}

class BBTLBExceptionIO extends Bundle {
  val interrupt = Output(Bool())
  val flush_retry = Input(Bool())
  val flush_skip = Input(Bool())

  def flush(dummy: Int = 0): Bool = flush_retry || flush_skip
}

class BBTLB(entries: Int, maxSize: Int)(implicit edge: TLEdgeOut, p: Parameters)
  extends CoreModule {

  val lgMaxSize = log2Ceil(maxSize)
  val io = IO(new Bundle {
    val req = Flipped(Valid(new BBTLBReq(lgMaxSize)))
    val resp = new TLBResp
    val ptw = new TLBPTWIO
    val exp = new BBTLBExceptionIO
  })

  val interrupt = RegInit(false.B)
  io.exp.interrupt := interrupt

  val tlb = Module(new TLB(false, lgMaxSize, TLBConfig(nSets=1, nWays=entries)))
  tlb.io.req.valid := io.req.valid
  tlb.io.req.bits := io.req.bits.tlb_req
  io.resp := tlb.io.resp
  tlb.io.kill := false.B

  tlb.io.sfence.valid     := io.exp.flush()
  tlb.io.sfence.bits.rs1  := false.B
  tlb.io.sfence.bits.rs2  := false.B
  tlb.io.sfence.bits.addr := DontCare
  tlb.io.sfence.bits.asid := DontCare
  tlb.io.sfence.bits.hv   := false.B
  tlb.io.sfence.bits.hg   := false.B

  io.ptw <> tlb.io.ptw
  tlb.io.ptw.status := io.req.bits.status

  val exception = io.req.valid && Mux(io.req.bits.tlb_req.cmd === M_XRD,
    tlb.io.resp.pf.ld || tlb.io.resp.ae.ld || tlb.io.resp.gf.ld,
    tlb.io.resp.pf.st || tlb.io.resp.ae.st || tlb.io.resp.gf.st)

  when (exception) {
    interrupt := true.B
  }

  when (interrupt && io.exp.flush_skip) {
    interrupt := false.B
  }

  when (interrupt && io.exp.flush_retry) {
    interrupt := false.B
  }

  assert(!io.exp.flush_retry || !io.exp.flush_skip, "TLB: flushing with both retry and skip at same time")
}
