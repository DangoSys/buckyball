package framework.memdomain.frontend.outside_channel.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.{MStatus, M_XWR}

import framework.utils.Util._
import framework.memdomain.frontend.outside_channel.tlb.BBTLBIO
import framework.top.GlobalConfig

class BBWriteRequest(dataWidth: Int) extends Bundle {
  val vaddr  = UInt(64.W)
  val data   = UInt(dataWidth.W)
  // Write length (bytes)
  val len    = UInt(16.W)
  // Byte mask
  val mask   = UInt((dataWidth / 8).W)
  val status = new MStatus
}

class BBWriteResponse extends Bundle {
  val done = Bool()
}

case class BBStreamWriterParam(
  nXacts:    Int,
  beatBits:  Int,
  maxBytes:  Int,
  dataWidth: Int,
  tledge:    TLEdgeOut)

@instantiable
class BBStreamWriter(val parameter: BBStreamWriterParam, val b: GlobalConfig) extends Module {

  val vaddrBits = b.core.vaddrBits

  @public
  val io = IO(new Bundle {
    val req   = Flipped(Decoupled(new BBWriteRequest(parameter.dataWidth)))
    val resp  = Decoupled(new BBWriteResponse)
    val tlb   = Flipped(new BBTLBIO(b))
    val busy  = Output(Bool())
    val flush = Input(Bool())
    // TileLink physical connection
    val tl    = new TLBundle(parameter.tledge.bundle)
  })

  val edge      = parameter.tledge
  val beatBytes = parameter.beatBits / 8

  val s_idle :: s_writing :: Nil = Enum(2)
  val state                      = RegInit(s_idle)

  val req = Reg(new BBWriteRequest(parameter.dataWidth))

  val xactBusy   = RegInit(0.U(parameter.nXacts.W))
  val xactOnehot = PriorityEncoderOH(~xactBusy)
  val xactId     = OHToUInt(xactOnehot)

  val xactBusy_fire   = WireInit(false.B)
  val xactBusy_add    = Mux(xactBusy_fire, (1.U << xactId).asUInt, 0.U)
  val xactBusy_remove = ~Mux(io.tlb.resp.valid && !io.tlb.resp.bits.miss, (1.U << xactId).asUInt, 0.U)
  xactBusy := (xactBusy | xactBusy_add) & xactBusy_remove.asUInt

  // Simplified: data is already aligned, directly construct TileLink request
  val lg_beat_bytes = log2Ceil(beatBytes)
  val use_put_full  = req.mask === ~0.U(beatBytes.W)

  val putFull = edge.Put(
    fromSource = xactId,
    toAddress = 0.U,
    lgSize = lg_beat_bytes.U,
    data = req.data
  )._2

  val putPartial = edge.Put(
    fromSource = xactId,
    toAddress = 0.U,
    lgSize = lg_beat_bytes.U,
    data = req.data,
    mask = req.mask
  )._2

  val selected_put = Mux(use_put_full, putFull, putPartial)

  // TLB processing pipeline
  class TLBundleAWithInfo extends Bundle {
    val tl_a   = selected_put.cloneType
    val vaddr  = Output(UInt(vaddrBits.W))
    val status = Output(new MStatus)
  }

  val untranslated_a = Wire(Decoupled(new TLBundleAWithInfo))
  xactBusy_fire              := untranslated_a.fire
  untranslated_a.valid       := state === s_writing && !xactBusy.andR
  untranslated_a.bits.tl_a   := selected_put
  untranslated_a.bits.vaddr  := req.vaddr
  untranslated_a.bits.status := req.status

  val tlb_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe = true))
  tlb_q.io.enq <> untranslated_a

  io.tlb.req.valid            := tlb_q.io.deq.valid
  io.tlb.req.bits             := DontCare
  io.tlb.req.bits.vaddr       := tlb_q.io.deq.bits.vaddr
  io.tlb.req.bits.passthrough := false.B
  io.tlb.req.bits.size        := 0.U
  io.tlb.req.bits.cmd         := M_XWR
  io.tlb.req.bits.prv         := 3.U // Machine mode
  io.tlb.req.bits.v           := false.B
  io.tlb.req.bits.status      := tlb_q.io.deq.bits.status

  val translate_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe = true))
  translate_q.io.enq <> tlb_q.io.deq
  translate_q.io.deq.ready := (io.tlb.resp.fire && !io.tlb.resp.bits.miss && io.tl.a.ready) || io.tlb.resp.bits.miss

  // TileLink A channel (request) connection
  io.tl.a.valid        := translate_q.io.deq.valid && !io.tlb.resp.bits.miss
  io.tl.a.bits         := translate_q.io.deq.bits.tl_a
  io.tl.a.bits.address := io.tlb.resp.bits.paddr

  // TileLink D channel (response) processing
  io.tl.d.ready := io.resp.ready

  io.resp.valid     := io.tl.d.valid
  io.resp.bits.done := true.B

  // Tie off unused TileLink channels
  io.tl.b.ready := true.B
  io.tl.c.valid := false.B
  io.tl.e.valid := false.B

  // State machine
  io.req.ready := state === s_idle
  io.busy      := xactBusy.orR || (state =/= s_idle)

  when(io.req.fire) {
    req   := io.req.bits
    state := s_writing
  }

  when(untranslated_a.fire) {
    state := s_idle
  }
}
