package framework.builtin.memdomain.dma

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy.{IdRange, LazyModule, LazyModuleImp}
import freechips.rocketchip.tile.{CoreBundle, HasCoreParameters}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.MStatus
import freechips.rocketchip.rocket.constants.MemoryOpConstants

import framework.builtin.util.Util._
import framework.builtin.memdomain.tlb.BBTLBIO
import framework.builtin.memdomain.dma.LocalAddr


class BBReadRequest()(implicit p: Parameters) extends CoreBundle {
  val vaddr = UInt(coreMaxAddrBits.W)
  // Read length (bytes)
  val len = UInt(16.W)
  val status = new MStatus
  // Stride (bytes)
  val stride = UInt(10.W)
}

class BBReadResponse(dataWidth: Int) extends Bundle {
  val data = UInt(dataWidth.W)
  val last = Bool()
  val addrcounter = UInt(10.W)
}

class BBWriteRequest(dataWidth: Int)(implicit p: Parameters) extends CoreBundle {
  val vaddr = UInt(coreMaxAddrBits.W)
  val data = UInt(dataWidth.W)
  // Write length (bytes)
  val len = UInt(16.W)
  // Byte mask
  val mask = UInt((dataWidth / 8).W)
  val status = new MStatus
}

class BBWriteResponse extends Bundle {
  val done = Bool()
}

class BBStreamReader(nXacts: Int, beatBits: Int, maxBytes: Int, dataWidth: Int)
                        (implicit p: Parameters) extends LazyModule {
  val node = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
    name = "buckyball-stream-reader", sourceId = IdRange(0, nXacts))))))

  lazy val module = new Impl
  class Impl extends LazyModuleImp(this) with HasCoreParameters with MemoryOpConstants {
    val (tl, edge) = node.out(0)
    val beatBytes = beatBits / 8

    val io = IO(new Bundle {
      val req = Flipped(Decoupled(new BBReadRequest()))
      val resp = Decoupled(new BBReadResponse(dataWidth))
      val tlb = new BBTLBIO
      val busy = Output(Bool())
      val flush = Input(Bool())
    })

    val s_idle :: s_req_new_block :: Nil = Enum(2)
    val state = RegInit(s_idle)

    val req = Reg(new BBReadRequest())
    // Number of bytes requested
    val bytesRequested = Reg(UInt(16.W))
    // Number of bytes received
    val bytesReceived = Reg(UInt(16.W))
    val bytesLeft = req.len - bytesRequested

    // Select request size - simplified version, fixed use of beatBytes
    val read_size = minOf(beatBytes.U, bytesLeft)
    val read_vaddr = req.vaddr + bytesRequested * req.stride

    // Track byte range corresponding to each request for correct last signal calculation
    // Starting byte position of current request
    val req_byte_start = Reg(UInt(16.W))
    // Ending byte position of current request
    val req_byte_end = Wire(UInt(16.W))
    req_byte_end := req_byte_start + read_size

    // Transaction ID management
    val xactBusy = RegInit(0.U(nXacts.W))
    val xactOnehot = PriorityEncoderOH(~xactBusy)
    val xactId = OHToUInt(xactOnehot)

    val xactBusy_fire = WireInit(false.B)
    val xactBusy_add = Mux(xactBusy_fire, (1.U << xactId).asUInt, 0.U)
    val xactBusy_remove = ~Mux(tl.d.fire, (1.U << tl.d.bits.source).asUInt, 0.U)
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
      val tl_a = tl.a.bits.cloneType
      val vaddr = Output(UInt(vaddrBits.W))
      val status = Output(new MStatus)
    }

    val untranslated_a = Wire(Decoupled(new TLBundleAWithInfo))
    xactBusy_fire := untranslated_a.fire && state === s_req_new_block
    untranslated_a.valid := state === s_req_new_block && !xactBusy.andR
    untranslated_a.bits.tl_a := get
    untranslated_a.bits.vaddr := read_vaddr
    untranslated_a.bits.status := req.status



    // Simplified: no retry mechanism, direct connection
    val tlb_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe=true))
    tlb_q.io.enq <> untranslated_a

    io.tlb.req.valid := tlb_q.io.deq.fire
    io.tlb.req.bits := DontCare
    io.tlb.req.bits.tlb_req.vaddr := tlb_q.io.deq.bits.vaddr
    io.tlb.req.bits.tlb_req.passthrough := false.B
    io.tlb.req.bits.tlb_req.size := 0.U
    io.tlb.req.bits.tlb_req.cmd := M_XRD
    io.tlb.req.bits.status := tlb_q.io.deq.bits.status

    val translate_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe=true))
    translate_q.io.enq <> tlb_q.io.deq
    translate_q.io.deq.ready := tl.a.ready || io.tlb.resp.miss

    // TileLink connection
    tl.a.valid := translate_q.io.deq.valid
    tl.a.bits := translate_q.io.deq.bits.tl_a
    tl.a.bits.address := io.tlb.resp.paddr

    // Iteration counter for tracking number of requests
    val iter_counter = RegInit(0.U(10.W))
    // Table for managing iteration counts
    val iter_mangage_table = RegInit(VecInit(Seq.fill(16)(0.U(10.W))))
    // Iteration count management - update after each request
    when (tl.a.fire) {
      iter_counter := iter_counter + 1.U
      iter_mangage_table(tl.a.bits.source) := iter_counter
    }

    // Response processing
    io.resp.valid := tl.d.valid
    io.resp.bits.data := tl.d.bits.data
    // Use source as address counter
    io.resp.bits.addrcounter := iter_mangage_table(tl.d.bits.source)
    // Fix last signal: calculate using received byte count
    // Total byte count after receiving current beat
    val resp_bytes_end = bytesReceived + beatBytes.U
    io.resp.bits.last := edge.last(tl.d) && (resp_bytes_end >= req.len)
    tl.d.ready := io.resp.ready

    // Update received byte count
    when (tl.d.fire) {
      bytesReceived := bytesReceived + beatBytes.U
    }

    // State machine
    io.req.ready := state === s_idle
    io.busy := xactBusy.orR || (state =/= s_idle)

    when (io.req.fire) {
      req := io.req.bits
      bytesRequested := 0.U
      // Reset received byte count
      bytesReceived := 0.U
      // Reset iteration counter
      iter_counter := 0.U
      state := s_req_new_block
    }

    when (untranslated_a.fire) {
      // Use actual requested byte count
      bytesRequested := bytesRequested + read_size
      // Check if more requests need to be sent
      when (bytesRequested + read_size >= req.len) {
        // All requests sent
        state := s_idle
      }.otherwise {
        // Continue sending next request
        state := s_req_new_block
      }
    }
  }
}

// Convention: data is already aligned and has mask
class BBStreamWriter(nXacts: Int, beatBits: Int, maxBytes: Int, dataWidth: Int)
                        (implicit p: Parameters) extends LazyModule {
  val node = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
    name = "buckyball-stream-writer", sourceId = IdRange(0, nXacts))))))

  lazy val module = new Impl
  class Impl extends LazyModuleImp(this) with HasCoreParameters with MemoryOpConstants {
    val (tl, edge) = node.out(0)
    val beatBytes = beatBits / 8

    val io = IO(new Bundle {
      val req = Flipped(Decoupled(new BBWriteRequest(dataWidth)))
      val resp = Decoupled(new BBWriteResponse)
      val tlb = new BBTLBIO
      val busy = Output(Bool())
      val flush = Input(Bool())
    })

    val s_idle :: s_writing :: Nil = Enum(2)
    val state = RegInit(s_idle)

    val req = Reg(new BBWriteRequest(dataWidth))

    val xactBusy = RegInit(0.U(nXacts.W))
    val xactOnehot = PriorityEncoderOH(~xactBusy)
    val xactId = OHToUInt(xactOnehot)

    val xactBusy_fire = WireInit(false.B)
    val xactBusy_add = Mux(xactBusy_fire, (1.U << xactId).asUInt, 0.U)
    val xactBusy_remove = ~Mux(tl.d.fire, (1.U << tl.d.bits.source).asUInt, 0.U)
    xactBusy := (xactBusy | xactBusy_add) & xactBusy_remove.asUInt

    // Simplified: data is already aligned, directly construct TileLink request
    val lg_beat_bytes = log2Ceil(beatBytes)
    val use_put_full = req.mask === ~0.U(beatBytes.W)

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
      val tl_a = tl.a.bits.cloneType
      val vaddr = Output(UInt(vaddrBits.W))
      val status = Output(new MStatus)
    }

    val untranslated_a = Wire(Decoupled(new TLBundleAWithInfo))
    xactBusy_fire := untranslated_a.fire
    untranslated_a.valid := state === s_writing && !xactBusy.andR
    untranslated_a.bits.tl_a := selected_put
    untranslated_a.bits.vaddr := req.vaddr
    untranslated_a.bits.status := req.status

    val tlb_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe=true))
    tlb_q.io.enq <> untranslated_a

    io.tlb.req.valid := tlb_q.io.deq.valid
    io.tlb.req.bits := DontCare
    io.tlb.req.bits.tlb_req.vaddr := tlb_q.io.deq.bits.vaddr
    io.tlb.req.bits.tlb_req.passthrough := false.B
    io.tlb.req.bits.tlb_req.size := 0.U
    io.tlb.req.bits.tlb_req.cmd := M_XWR
    io.tlb.req.bits.status := tlb_q.io.deq.bits.status

    val translate_q = Module(new Queue(new TLBundleAWithInfo, 1, pipe=true))
    translate_q.io.enq <> tlb_q.io.deq
    translate_q.io.deq.ready := tl.a.ready || io.tlb.resp.miss

    // TileLink connection
    tl.a.valid := translate_q.io.deq.valid && !io.tlb.resp.miss
    tl.a.bits := translate_q.io.deq.bits.tl_a
    tl.a.bits.address := io.tlb.resp.paddr

    tl.d.ready := true.B

    // Response processing
    io.resp.valid := tl.d.valid && edge.last(tl.d)
    io.resp.bits.done := true.B

    // State machine
    io.req.ready := state === s_idle
    io.busy := xactBusy.orR || (state =/= s_idle)

    when (io.req.fire) {
      req := io.req.bits
      state := s_writing
    }

    when (untranslated_a.fire) {
      state := s_idle
    }
  }
}
