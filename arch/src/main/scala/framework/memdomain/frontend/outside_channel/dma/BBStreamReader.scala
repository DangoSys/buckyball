package framework.memdomain.frontend.outside_channel.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.{MStatus, M_XRD}
import freechips.rocketchip.rocket.constants.MemoryOpConstants

import framework.utils.Util._
import framework.memdomain.frontend.outside_channel.tlb.BBTLBIO
import framework.top.GlobalConfig

class BBReadRequest extends Bundle {
  val vaddr  = UInt(64.W)
  // Read length (bytes)
  val len    = UInt(16.W)
  val status = new MStatus
  // Stride (bytes)
  val stride = UInt(10.W)
}

class BBReadResponse(dataWidth: Int) extends Bundle {
  val data        = UInt(dataWidth.W)
  val last        = Bool()
  val addrcounter = UInt(10.W)
}

case class BBStreamReaderParam(
  nXacts:    Int,
  beatBits:  Int,
  maxBytes:  Int,
  dataWidth: Int,
  tledge:    TLEdgeOut)

@instantiable
class BBStreamReader(val parameter: BBStreamReaderParam, val b: GlobalConfig) extends Module {

  val vaddrBits = b.core.vaddrBits

  @public
  val io = IO(new Bundle {
    val req   = Flipped(Decoupled(new BBReadRequest()))
    val resp  = Decoupled(new BBReadResponse(parameter.dataWidth))
    val tlb   = Flipped(new BBTLBIO(b))
    val busy  = Output(Bool())
    val flush = Input(Bool())
    // TileLink physical connection
    val tl    = new TLBundle(parameter.tledge.bundle)
  })

  val edge      = parameter.tledge
  val beatBytes = parameter.beatBits / 8

  val s_idle :: s_req_new_block :: Nil = Enum(2)
  val state                            = RegInit(s_idle)

  val req            = Reg(new BBReadRequest())
  // Number of bytes requested
  val bytesRequested = Reg(UInt(16.W))
  // Number of bytes received
  val bytesReceived  = Reg(UInt(16.W))
  val bytesLeft      = req.len - bytesRequested

  // Select request size - simplified version, fixed use of beatBytes
  val read_size  = minOf(beatBytes.U, bytesLeft)
  val read_vaddr = req.vaddr + bytesRequested * req.stride

  // Track byte range corresponding to each request for correct last signal calculation
  // Starting byte position of current request
  val req_byte_start = Reg(UInt(16.W))
  // Ending byte position of current request
  val req_byte_end   = Wire(UInt(16.W))
  req_byte_end := req_byte_start + read_size

  // Transaction ID management
  val xactBusy   = RegInit(0.U(parameter.nXacts.W))
  val xactOnehot = PriorityEncoderOH(~xactBusy)
  val xactId     = OHToUInt(xactOnehot)

  val xactBusy_fire   = WireInit(false.B)
  val xactBusy_add    = Mux(xactBusy_fire, (1.U << xactId).asUInt, 0.U)
  val xactBusy_remove = ~Mux(io.tlb.resp.valid && !io.tlb.resp.bits.miss, (1.U << xactId).asUInt, 0.U)
  xactBusy := (xactBusy | xactBusy_add) & xactBusy_remove.asUInt

  // TileLink request construction - return to single beat requests to avoid address alignment issues
  val get = edge.Get(
    fromSource = xactId,
    toAddress = 0.U,
    // Request only one beat each time
    lgSize = log2Ceil(beatBytes).U
  )._2

  // TLB processing pipeline - simplified based on Gemmini
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

  // Simplified: no retry mechanism, direct connection
  val tlb_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe = true))
  tlb_q.io.enq <> untranslated_a

  io.tlb.req.valid            := tlb_q.io.deq.fire
  io.tlb.req.bits             := DontCare
  io.tlb.req.bits.vaddr       := tlb_q.io.deq.bits.vaddr
  io.tlb.req.bits.passthrough := false.B
  io.tlb.req.bits.size        := 0.U
  io.tlb.req.bits.cmd         := M_XRD
  io.tlb.req.bits.prv         := 3.U // Machine mode
  io.tlb.req.bits.v           := false.B
  io.tlb.req.bits.status      := tlb_q.io.deq.bits.status

  val translate_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe = true))
  translate_q.io.enq <> tlb_q.io.deq
  translate_q.io.deq.ready := io.tlb.resp.fire || io.tlb.resp.bits.miss

  // TileLink A channel (request) connection
  io.tl.a.valid            := translate_q.io.deq.valid && !io.tlb.resp.bits.miss
  io.tl.a.bits             := translate_q.io.deq.bits.tl_a
  io.tl.a.bits.address     := io.tlb.resp.bits.paddr
  translate_q.io.deq.ready := (io.tlb.resp.fire && !io.tlb.resp.bits.miss && io.tl.a.ready) || io.tlb.resp.bits.miss

  // Iteration counter for tracking number of requests
  val iter_counter       = RegInit(0.U(10.W))
  // Table for managing iteration counts
  val iter_mangage_table = RegInit(VecInit(Seq.fill(16)(0.U(10.W))))
  // Iteration count management - update after each request
  when(io.tl.a.fire) {
    iter_counter               := iter_counter + 1.U
    iter_mangage_table(xactId) := iter_counter
  }

  // TileLink D channel (response) processing
  io.tl.d.ready := io.resp.ready

  io.resp.valid            := io.tl.d.valid
  io.resp.bits.data        := io.tl.d.bits.data
  // Use source as address counter
  io.resp.bits.addrcounter := iter_mangage_table(io.tl.d.bits.source)
  // Fix last signal: calculate using received byte count
  // Total byte count after receiving current beat
  val resp_bytes_end = bytesReceived + beatBytes.U
  io.resp.bits.last := io.tl.d.valid && (resp_bytes_end >= req.len)

  // Update received byte count
  when(io.tl.d.fire) {
    bytesReceived := bytesReceived + beatBytes.U
  }

  // Tie off unused TileLink channels
  io.tl.b.ready := true.B
  io.tl.c.valid := false.B
  io.tl.e.valid := false.B

  // State machine
  io.req.ready := state === s_idle
  io.busy      := xactBusy.orR || (state =/= s_idle)

  when(io.req.fire) {
    req            := io.req.bits
    bytesRequested := 0.U
    // Reset received byte count
    bytesReceived  := 0.U
    // Reset iteration counter
    iter_counter   := 0.U
    state          := s_req_new_block
  }

  when(untranslated_a.fire) {
    // Use actual requested byte count
    bytesRequested := bytesRequested + read_size
    // Check if more requests need to be sent
    when(bytesRequested + read_size >= req.len) {
      // All requests sent
      state := s_idle
    }.otherwise {
      // Continue sending next request
      state := s_req_new_block
    }
  }
}
