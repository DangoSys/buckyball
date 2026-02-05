package framework.memdomain.frontend.outside_channel.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.{MStatus, M_XWR}
import framework.memdomain.frontend.outside_channel.tlb.BBTLBIO
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
class StreamWriter(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {

  val vaddrBits = b.core.vaddrBits
  val beatBits  = b.memDomain.dma_buswidth
  val dataWidth = b.memDomain.dma_buswidth
  val beatBytes = beatBits / 8
  val lgBeat    = log2Ceil(beatBytes)

  @public
  val io = IO(new Bundle {
    val req   = Flipped(Decoupled(new BBWriteRequest(dataWidth)))
    val resp  = Decoupled(new BBWriteResponse)
    val tlb   = Flipped(new BBTLBIO(b))
    val busy  = Output(Bool())
    val flush = Input(Bool())
    val tl    = new TLBundle(edge.bundle)
  })

  // ---------------------------------------------------------------------------
  // Strict single-outstanding writer with PROPER handshakes
  // ---------------------------------------------------------------------------
  // NOTE: current TLB/Cluster returns resp combinationally in the same cycle as req.valid
  // (see StreamReader usage). So we must NOT "fire req then wait for resp".
  val s_idle :: s_tlb_req :: s_wait_d :: s_resp :: Nil = Enum(4)
  val state                                            = RegInit(s_idle)

  val reqReg = Reg(new BBWriteRequest(dataWidth))

  // single outstanding => fixed source id 0
  val xactId = 0.U(io.tl.a.bits.source.getWidth.W)

  // -----------------------
  // Accept one request
  // -----------------------
  io.req.ready := (state === s_idle)

  when(io.req.fire) {
    reqReg := io.req.bits
    state  := s_tlb_req
  }

  // -----------------------
  // Construct TileLink Put from LATCHED request
  // -----------------------
  val use_put_full = reqReg.mask === ~0.U(beatBytes.W)

  val putFull = edge.Put(
    fromSource = xactId,
    toAddress = 0.U, // overwritten later
    lgSize = lgBeat.U,
    data = reqReg.data
  )._2

  val putPartial = edge.Put(
    fromSource = xactId,
    toAddress = 0.U, // overwritten later
    lgSize = lgBeat.U,
    data = reqReg.data,
    mask = reqReg.mask
  )._2

  val putMsg = Wire(putFull.cloneType)
  putMsg := Mux(use_put_full, putFull, putPartial)

  // -----------------------
  // TLB handshake (req.fire -> wait resp.valid)
  // -----------------------
  io.tlb.req.valid            := (state === s_tlb_req)
  io.tlb.req.bits             := DontCare
  io.tlb.req.bits.vaddr       := reqReg.vaddr
  io.tlb.req.bits.passthrough := true.B
  io.tlb.req.bits.size        := 0.U
  io.tlb.req.bits.cmd         := M_XWR
  io.tlb.req.bits.prv         := 3.U
  io.tlb.req.bits.v           := false.B
  io.tlb.req.bits.status      := reqReg.status

  // We only "consume" the tlb response when we actually send A.
  // This matches StreamReader: resp.valid is treated as a combinational translate result.
  io.tlb.resp.ready := (state === s_tlb_req) && io.tl.a.ready

  // -----------------------
  // TileLink A channel (only when tlb resp is present)
  // -----------------------
  io.tl.a.valid        := (state === s_tlb_req) && io.tlb.resp.valid
  io.tl.a.bits         := putMsg
  io.tl.a.bits.address := io.tlb.resp.bits.paddr

  when(state === s_tlb_req && io.tl.a.fire) {
    state := s_wait_d
  }

  // -----------------------
  // TileLink D channel (ack)
  // -----------------------
  io.tl.d.ready := (state === s_wait_d)

  // upper response
  io.resp.valid     := (state === s_resp)
  io.resp.bits.done := true.B

  when(state === s_wait_d && io.tl.d.fire) {
    state := s_resp
  }

  when(state === s_resp && io.resp.fire) {
    state := s_idle
  }

  // -----------------------
  // Tie off unused TL channels
  // -----------------------
  io.tl.b.ready := true.B
  io.tl.c.valid := false.B
  io.tl.e.valid := false.B

  io.busy := (state =/= s_idle)
}
