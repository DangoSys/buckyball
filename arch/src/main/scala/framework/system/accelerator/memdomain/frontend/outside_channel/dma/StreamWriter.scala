package framework.system.accelerator.memdomain.frontend.outside_channel.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.amba.axi4.{AXI4Bundle, AXI4BundleParameters}
import freechips.rocketchip.rocket.{MStatus, M_XWR}
import framework.system.accelerator.memdomain.frontend.outside_channel.tlb.BBTLBIO
import framework.system.link.BBAxi4Params
import framework.top.GlobalConfig

class BBWriteRequest(dataWidth: Int) extends Bundle {
  val vaddr  = UInt(64.W)
  val data   = UInt(dataWidth.W)
  val len    = UInt(16.W)
  val mask   = UInt((dataWidth / 8).W)
  val status = new MStatus
}

class BBWriteResponse extends Bundle {
  val done = Bool()
}

@instantiable
class StreamWriter(val b: GlobalConfig) extends Module {

  val vaddrBits = b.core.vaddrBits
  val beatBits  = b.memDomain.dma_buswidth
  val dataWidth = b.memDomain.dma_buswidth
  val beatBytes = beatBits / 8
  val lgBeat    = log2Ceil(beatBytes)
  val axiParams: AXI4BundleParameters = BBAxi4Params(b)

  @public
  val io = IO(new Bundle {
    val req   = Flipped(Decoupled(new BBWriteRequest(dataWidth)))
    val resp  = Decoupled(new BBWriteResponse)
    val tlb   = Flipped(new BBTLBIO(b))
    val busy  = Output(Bool())
    val flush = Input(Bool())
    val axi   = AXI4Bundle(axiParams)
  })

  // ---------------------------------------------------------------------------
  // Strict single-outstanding writer with PROPER handshakes
  // ---------------------------------------------------------------------------
  // For AXI4: AW + W must both be issued for a single beat write. We assert
  // both together once the TLB returns a valid translation, then wait for the
  // B-channel acknowledgement before reporting completion upstream.
  val s_idle :: s_tlb_req :: s_wait_b :: s_resp :: Nil = Enum(4)
  val state                                            = RegInit(s_idle)

  val reqReg = Reg(new BBWriteRequest(dataWidth))

  // single outstanding => fixed transaction id 0
  val xactId = 0.U(axiParams.idBits.W)

  // Track which sub-channels (AW / W) have already fired so we can hold the
  // remaining one until both complete.
  val awSent = RegInit(false.B)
  val wSent  = RegInit(false.B)

  // -----------------------
  // Accept one request
  // -----------------------
  io.req.ready := (state === s_idle)

  when(io.req.fire) {
    reqReg := io.req.bits
    awSent := false.B
    wSent  := false.B
    state  := s_tlb_req
  }

  // -----------------------
  // TLB handshake (combinational translation, like StreamReader)
  // -----------------------
  io.tlb.req.valid            := (state === s_tlb_req)
  io.tlb.req.bits             := DontCare
  io.tlb.req.bits.vaddr       := reqReg.vaddr
  io.tlb.req.bits.passthrough := false.B
  io.tlb.req.bits.size        := 0.U
  io.tlb.req.bits.cmd         := M_XWR
  io.tlb.req.bits.prv         := 3.U
  io.tlb.req.bits.v           := false.B
  io.tlb.req.bits.status      := reqReg.status

  // We "consume" the tlb response only when the AXI request has fully launched.
  // For a single-beat write, this is when both AW and W have fired (or are
  // firing in this cycle).
  val tlbOk        = io.tlb.resp.valid && !io.tlb.resp.bits.miss
  val awCanFire    = (state === s_tlb_req) && tlbOk && !awSent
  val wCanFire     = (state === s_tlb_req) && tlbOk && !wSent
  val awFireNow    = awCanFire && io.axi.aw.ready
  val wFireNow     = wCanFire && io.axi.w.ready
  val awComplete   = awSent || awFireNow
  val wComplete    = wSent || wFireNow
  val xactLaunched = awComplete && wComplete

  io.tlb.resp.ready := (state === s_tlb_req) && xactLaunched

  // -----------------------
  // AXI4 AW channel
  // -----------------------
  io.axi.aw.valid      := awCanFire
  io.axi.aw.bits       := DontCare
  io.axi.aw.bits.id    := xactId
  io.axi.aw.bits.addr  := io.tlb.resp.bits.paddr
  io.axi.aw.bits.len   := 0.U // single beat
  io.axi.aw.bits.size  := lgBeat.U
  io.axi.aw.bits.burst := 1.U // INCR
  io.axi.aw.bits.lock  := 0.U
  io.axi.aw.bits.cache := 0.U
  io.axi.aw.bits.prot  := 0.U
  io.axi.aw.bits.qos   := 0.U

  // -----------------------
  // AXI4 W channel
  // -----------------------
  io.axi.w.valid     := wCanFire
  io.axi.w.bits      := DontCare
  io.axi.w.bits.data := reqReg.data
  io.axi.w.bits.strb := reqReg.mask
  io.axi.w.bits.last := true.B // single beat -> always last

  when(awFireNow)(awSent := true.B)
  when(wFireNow)(wSent   := true.B)

  when(state === s_tlb_req && xactLaunched) {
    state := s_wait_b
  }

  // -----------------------
  // AXI4 B channel (write ack)
  // -----------------------
  io.axi.b.ready := (state === s_wait_b)

  when(state === s_wait_b && io.axi.b.fire) {
    awSent := false.B
    wSent  := false.B
    state  := s_resp
  }

  // upper response
  io.resp.valid     := (state === s_resp)
  io.resp.bits.done := true.B

  when(state === s_resp && io.resp.fire) {
    state := s_idle
  }

  // -----------------------
  // Tie off unused AXI4 channels (read path)
  // -----------------------
  io.axi.ar.valid := false.B
  io.axi.ar.bits  := DontCare
  io.axi.r.ready  := true.B

  io.busy := (state =/= s_idle)
}
