package framework.memdomain.frontend.outside_channel.tlb

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile.{CoreBundle, CoreModule}
import freechips.rocketchip.tilelink.TLEdgeOut

class BBTLBIO(implicit p: Parameters) extends CoreBundle {
  val lgMaxSize = log2Ceil(coreDataBytes)
  val req       = Valid(new BBTLBReq(lgMaxSize))
  val resp      = Flipped(new TLBResp)
}

class BBTLBCluster(
  nClients:          Int,
  entries:           Int,
  maxSize:           Int,
  use_shared_tlb:    Boolean = true
)(
  implicit val edge: TLEdgeOut,
  p:                 Parameters)
    extends CoreModule {

  val num_tlbs  = if (use_shared_tlb) 1 else nClients
  val lgMaxSize = log2Ceil(coreDataBytes)

  val io = IO(new Bundle {
    val clients = Flipped(Vec(nClients, new BBTLBIO))
    val ptw     = Vec(num_tlbs, new TLBPTWIO)
    val exp     = Vec(num_tlbs, new BBTLBExceptionIO)
  })

  val tlbs = Seq.fill(num_tlbs)(Module(new BBTLB(entries, maxSize)))

  io.ptw <> VecInit(tlbs.map(_.io.ptw))
  io.exp <> VecInit(tlbs.map(_.io.exp))

  val tlbArbOpt = if (use_shared_tlb) Some(Module(new RRArbiter(new BBTLBReq(lgMaxSize), nClients))) else None

  if (use_shared_tlb) {
    val tlbArb = tlbArbOpt.get
    val tlb    = tlbs.head
    tlb.io.req.valid    := tlbArb.io.out.valid
    tlb.io.req.bits     := tlbArb.io.out.bits
    tlbArb.io.out.ready := true.B
  }

  io.clients.zipWithIndex.foreach {
    case (client, i) =>
      val last_translated_valid = RegInit(false.B)
      val last_translated_vpn   = RegInit(0.U(vaddrBits.W))
      val last_translated_ppn   = RegInit(0.U(paddrBits.W))

      val l0_tlb_hit   =
        last_translated_valid && ((client.req.bits.tlb_req.vaddr >> pgIdxBits).asUInt === (last_translated_vpn >> pgIdxBits).asUInt)
      val l0_tlb_paddr = Cat(last_translated_ppn >> pgIdxBits, client.req.bits.tlb_req.vaddr(pgIdxBits - 1, 0))

      val tlb         = if (use_shared_tlb) tlbs.head else tlbs(i)
      val tlbReq      = if (use_shared_tlb) tlbArbOpt.get.io.in(i).bits else tlb.io.req.bits
      val tlbReqValid = if (use_shared_tlb) tlbArbOpt.get.io.in(i).valid else tlb.io.req.valid
      val tlbReqFire  = if (use_shared_tlb) tlbArbOpt.get.io.in(i).fire else tlb.io.req.fire

      val l0_tlb_paddr_reg = RegEnable(client.req.bits.tlb_req.vaddr, client.req.valid)

      tlbReqValid := RegNext(client.req.valid && !l0_tlb_hit)
      tlbReq      := RegNext(client.req.bits)

      when(tlbReqFire && !tlb.io.resp.miss) {
        last_translated_valid := true.B
        last_translated_vpn   := tlbReq.tlb_req.vaddr
        last_translated_ppn   := tlb.io.resp.paddr
      }

      when(tlb.io.exp.flush()) {
        last_translated_valid := false.B
      }

      when(tlbReqFire) {
        client.resp := tlb.io.resp
      }.otherwise {
        client.resp       := DontCare
        client.resp.paddr := RegNext(l0_tlb_paddr)
        client.resp.miss  := !RegNext(l0_tlb_hit)
      }
  }
}
