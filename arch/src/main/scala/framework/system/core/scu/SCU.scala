package framework.system.core.scu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.tilelink._

/**
 * SCU (Synchronization Control Unit)
 *
 * Sits inside each Core between the TLXbar and the Core's external TL output.
 * Intercepts MMIO traffic (within configured address range) and handles it
 * internally, while forwarding non-MMIO traffic transparently.
 *
 * Two modes:
 * - interceptMMIO=false: Pure passthrough
 * - interceptMMIO=true: Intercept addresses in [mmioBase, mmioBase + mmioSize)
 *
 * Internal MMIO handling (simple stub for now):
 * - Reads (Get): return zero-filled data
 * - Writes (PutFull/PutPartial): return AccessAck
 *
 * Future enhancements:
 * - Connect to actual MMIO devices (registers, status, etc.)
 * - Synchronization primitives (barriers, fences)
 * - Performance counters
 */
@instantiable
class SCU(
  val params:         SCUParams,
  val tlBundleParams: TLBundleParameters)
    extends Module {

  @public
  val io = IO(new Bundle {
    val tl_in  = Flipped(new TLBundle(tlBundleParams))
    val tl_out = new TLBundle(tlBundleParams)
  })

  if (!params.interceptMMIO) {
    // Pure passthrough mode
    io.tl_out <> io.tl_in
  } else {
    // === MMIO interception mode ===

    // Check if A channel address is in MMIO range
    val a_addr  = io.tl_in.a.bits.address
    val is_mmio = (a_addr >= params.mmioBase.U) && (a_addr < (params.mmioBase + params.mmioSize).U)

    // === A channel routing ===
    // Non-MMIO: forward to outer
    io.tl_out.a.valid := io.tl_in.a.valid && !is_mmio
    io.tl_out.a.bits  := io.tl_in.a.bits

    // MMIO: capture into internal queue for response generation
    val mmio_req_q = Module(new Queue(io.tl_in.a.bits.cloneType, 4))
    mmio_req_q.io.enq.valid := io.tl_in.a.valid && is_mmio
    mmio_req_q.io.enq.bits  := io.tl_in.a.bits

    // A channel ready: combined from outer (non-MMIO) or queue (MMIO)
    io.tl_in.a.ready := Mux(is_mmio, mmio_req_q.io.enq.ready, io.tl_out.a.ready)

    // === D channel routing ===
    // Generate MMIO response from queue
    val mmio_resp = Wire(new TLBundleD(tlBundleParams))
    mmio_resp         := DontCare
    mmio_resp.opcode  := Mux(
      mmio_req_q.io.deq.bits.opcode === TLMessages.Get,
      TLMessages.AccessAckData,
      TLMessages.AccessAck
    )
    mmio_resp.param   := 0.U
    mmio_resp.size    := mmio_req_q.io.deq.bits.size
    mmio_resp.source  := mmio_req_q.io.deq.bits.source
    mmio_resp.sink    := 0.U
    mmio_resp.denied  := false.B
    mmio_resp.data    := 0.U // Always return zeros for MMIO reads (stub)
    mmio_resp.corrupt := false.B

    // Arbitrate D channel: prioritize outer responses, then internal MMIO
    val outer_d_valid = io.tl_out.d.valid
    val mmio_d_valid  = mmio_req_q.io.deq.valid

    io.tl_in.d.valid := outer_d_valid || mmio_d_valid
    io.tl_in.d.bits  := Mux(outer_d_valid, io.tl_out.d.bits, mmio_resp)

    io.tl_out.d.ready       := io.tl_in.d.ready
    mmio_req_q.io.deq.ready := io.tl_in.d.ready && !outer_d_valid

    // === B/C/E channels (TL-UH: tied off) ===
    io.tl_in.b.valid := false.B
    io.tl_in.b.bits  := DontCare
    io.tl_in.c.ready := true.B
    io.tl_in.e.ready := true.B

    io.tl_out.b.ready := true.B
    io.tl_out.c.valid := false.B
    io.tl_out.c.bits  := DontCare
    io.tl_out.e.valid := false.B
    io.tl_out.e.bits  := DontCare
  }
}
