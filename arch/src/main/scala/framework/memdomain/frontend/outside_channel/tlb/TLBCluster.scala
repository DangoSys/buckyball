package framework.memdomain.frontend.outside_channel.tlb

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tilelink.TLEdgeOut
import framework.top.GlobalConfig

@instantiable
class BBTLBCluster(val b: GlobalConfig)(implicit val edge: TLEdgeOut) extends Module {

  val nClients  = 2
  val entries   = b.memDomain.tlb_size
  val maxSize   = b.core.coreDataBytes
  val lgMaxSize = log2Ceil(b.core.coreDataBytes)
  val vaddrBits = b.core.vaddrBits
  val paddrBits = b.core.paddrBits
  val pgIdxBits = b.core.pgIdxBits

  @public
  val io = IO(new Bundle {
    val clients = Vec(nClients, new BBTLBIO(b))
    val ptw     = Vec(1, new BBTLBPTWIO(b))    // Shared TLB has only 1 PTW port
    val exp     = Vec(1, new BBTLBExceptionIO) // Shared TLB has only 1 exception interface
  })

  val tlb = Instantiate(new TLB(b, lgMaxSize))

  // Exception handling
  val interrupt = RegInit(false.B)
  io.exp(0).interrupt := interrupt

  // Connect PTW
  io.ptw(0) <> tlb.io.ptw

  val tlbArb    = Module(new RRArbiter(new BBTLBReq(lgMaxSize, vaddrBits, b.core.xLen), nClients))
  val tlbArbOut = tlbArb.io.out
  val tlb_io    = tlb.io

  tlb_io.req.valid := tlbArbOut.valid
  tlb_io.req.bits  := tlbArbOut.bits
  tlbArbOut.ready  := tlb_io.req.ready

  // Connect status to PTW
  tlb_io.ptw.status := tlbArbOut.bits.status
  tlb_io.kill       := false.B

  // Handle sfence from exception IO
  tlb_io.sfence.valid := io.exp(0).flush()
  tlb_io.sfence.bits  := false.B

  // Exception detection
  val isRead = tlbArbOut.bits.cmd(0) === 0.U

  val exception = tlbArbOut.valid && !tlb_io.resp.bits.miss && Mux(
    isRead,
    tlb_io.resp.bits.pf.ld || tlb_io.resp.bits.ae.ld || tlb_io.resp.bits.gf.ld,
    tlb_io.resp.bits.pf.st || tlb_io.resp.bits.ae.st || tlb_io.resp.bits.gf.st
  )

  when(exception) {
    interrupt := true.B
  }

  when(interrupt && io.exp(0).flush_skip) {
    interrupt := false.B
  }

  when(interrupt && io.exp(0).flush_retry) {
    interrupt := false.B
  }

  assert(!io.exp(0).flush_retry || !io.exp(0).flush_skip, "TLB: flushing with both retry and skip at same time")

  io.clients.zipWithIndex.foreach {
    case (client, i) =>
      val last_translated_valid = RegInit(false.B)
      val last_translated_vpn   = RegInit(0.U(vaddrBits.W))
      val last_translated_ppn   = RegInit(0.U(paddrBits.W))

      val l0_tlb_hit   =
        last_translated_valid && ((client.req.bits.vaddr >> pgIdxBits).asUInt === (last_translated_vpn >> pgIdxBits).asUInt)
      val l0_tlb_paddr = Cat(last_translated_ppn >> pgIdxBits, client.req.bits.vaddr(pgIdxBits - 1, 0))

      tlbArb.io.in(i).valid := client.req.valid
      tlbArb.io.in(i).bits  := client.req.bits
      client.req.ready      := tlbArb.io.in(i).ready

      val tlbReq     = tlbArb.io.in(i).bits
      val tlbReqFire = tlbArb.io.in(i).fire

      when(tlbReqFire && !tlb_io.resp.bits.miss) {
        last_translated_valid := true.B
        last_translated_vpn   := tlbReq.vaddr
        last_translated_ppn   := tlb_io.resp.bits.paddr
      }

      when(io.exp(0).flush()) {
        last_translated_valid := false.B
      }
      client.resp.bits.miss := tlb_io.resp.bits.miss
      when(client.req.valid || tlb.io.resp.valid) {
        client.resp <> tlb_io.resp
      }.otherwise {
        client.resp.valid := false.B
        client.resp.bits  := DontCare
        tlb_io.resp.ready := false.B
      }
  }
}
