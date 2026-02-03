package framework.memdomain.frontend.outside_channel.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.{MStatus, M_XRD}

import framework.builtin.utils.Util._
import framework.memdomain.frontend.outside_channel.tlb.BBTLBIO
import framework.top.GlobalConfig

class BBReadRequest extends Bundle {
  val vaddr  = UInt(64.W)
  val len    = UInt(16.W)
  val status = new MStatus
  val stride = UInt(10.W)
}

class BBReadResponse(dataWidth: Int) extends Bundle {
  val data        = UInt(dataWidth.W)
  val last        = Bool()
  val addrcounter = UInt(10.W) // will be beat index: 0,1,2,...
}

@instantiable
class StreamReader(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {
  val vaddrBits = b.core.vaddrBits
  val nXacts    = b.memDomain.dma_n_xacts
  val beatBits  = b.memDomain.dma_buswidth
  val maxBytes  = b.memDomain.dma_maxbytes
  val dataWidth = b.memDomain.dma_buswidth
  val beatBytes = beatBits / 8

  @public
  val io = IO(new Bundle {
    val req   = Flipped(Decoupled(new BBReadRequest()))
    val resp  = Decoupled(new BBReadResponse(dataWidth))
    val tlb   = Flipped(new BBTLBIO(b))
    val busy  = Output(Bool())
    val flush = Input(Bool())
    val tl    = new TLBundle(edge.bundle)
  })

  val s_idle :: s_req_new_block :: Nil = Enum(2)
  val state                            = RegInit(s_idle)

  val req = Reg(new BBReadRequest())

  // Number of bytes requested (A sent)
  val bytesRequested = RegInit(0.U(16.W))
  // Number of bytes received (D accepted)
  val bytesReceived  = RegInit(0.U(16.W))

  val bytesLeft = req.len - bytesRequested

  // Select request size - simplified version, fixed use of beatBytes
  val read_size  = minOf(beatBytes.U, bytesLeft)
  val read_vaddr = req.vaddr + bytesRequested * req.stride

  // Transaction ID management
  val xactBusy   = RegInit(0.U(nXacts.W))
  val xactOnehot = PriorityEncoderOH(~xactBusy)
  val xactId     = OHToUInt(xactOnehot)

  val xactBusy_fire   = WireInit(false.B)
  val xactBusy_add    = Mux(xactBusy_fire, (1.U << xactId).asUInt, 0.U)
  val xactBusy_remove = ~Mux(io.tl.d.fire, (1.U << io.tl.d.bits.source).asUInt, 0.U)
  xactBusy := (xactBusy | xactBusy_add) & xactBusy_remove.asUInt

  // TileLink request construction - single beat
  val get = edge.Get(
    fromSource = xactId,
    toAddress = 0.U,
    lgSize = log2Ceil(beatBytes).U
  )._2

  // TLB processing pipeline
  class TLBundleAWithInfo extends Bundle {
    val tl_a   = get.cloneType
    val vaddr  = Output(UInt(vaddrBits.W))
    val status = Output(new MStatus)
  }

  val untranslated_a = Wire(Decoupled(new TLBundleAWithInfo))
  xactBusy_fire              := untranslated_a.fire && state === s_req_new_block
  untranslated_a.valid       := state === s_req_new_block && !xactBusy.andR
  untranslated_a.bits.tl_a   := get
  untranslated_a.bits.vaddr  := read_vaddr
  untranslated_a.bits.status := req.status

  // 1-deep queue to break comb paths
  val tlb_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe = true))
  tlb_q.io.enq <> untranslated_a

  io.tlb.req.valid            := tlb_q.io.deq.valid && (state === s_req_new_block)
  io.tlb.req.bits             := DontCare
  io.tlb.req.bits.vaddr       := tlb_q.io.deq.bits.vaddr
  io.tlb.req.bits.passthrough := true.B
  io.tlb.req.bits.size        := 0.U
  io.tlb.req.bits.cmd         := M_XRD
  io.tlb.req.bits.prv         := 3.U
  io.tlb.req.bits.v           := false.B
  io.tlb.req.bits.status      := tlb_q.io.deq.bits.status

  // consume tlb_q when tlb resp valid
  tlb_q.io.deq.ready := io.tlb.resp.valid

  // TileLink A channel
  io.tl.a.valid        := io.tlb.resp.valid && (state === s_req_new_block)
  io.tl.a.bits         := tlb_q.io.deq.bits.tl_a
  io.tl.a.bits.address := io.tlb.resp.bits.paddr
  io.tlb.resp.ready    := true.B

  // -----------------------------
  // D channel -> resp
  // -----------------------------
  // Backpressure from resp to TileLink D
  io.tl.d.ready := io.resp.ready

  io.resp.valid     := io.tl.d.valid
  io.resp.bits.data := io.tl.d.bits.data

  // ✅ FIX: addrcounter = beat index in receive order (0,1,2,...)
  // Current beat index before consuming this beat:
  val beatIndex = (bytesReceived >> log2Ceil(beatBytes)).asUInt
  io.resp.bits.addrcounter := beatIndex(9, 0)

  // last flag based on total bytes after this beat
  val resp_bytes_end = bytesReceived + beatBytes.U
  io.resp.bits.last := io.tl.d.valid && (resp_bytes_end >= req.len)

  // Update received bytes when we actually accept a beat
  when(io.tl.d.fire) {
    bytesReceived := bytesReceived + beatBytes.U
  }

  // Tie off unused TileLink channels
  io.tl.b.ready := true.B
  io.tl.c.valid := false.B
  io.tl.e.valid := false.B

  // -----------------------------
  // State machine
  // -----------------------------
  io.req.ready := (state === s_idle)
  io.busy      := xactBusy.orR || (state =/= s_idle)

  when(io.req.fire) {
    req            := io.req.bits
    bytesRequested := 0.U
    bytesReceived  := 0.U
    state          := s_req_new_block
  }

  // IMPORTANT: bytesRequested should advance only when A actually fires
  when(io.tl.a.fire) {
    val nextBytesRequested = bytesRequested + read_size
    bytesRequested := nextBytesRequested

    when(nextBytesRequested >= req.len) {
      state := s_idle
    }.otherwise {
      state := s_req_new_block
    }
  }
}
