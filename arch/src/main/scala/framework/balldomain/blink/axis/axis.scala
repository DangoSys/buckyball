package framework.balldomain.blink.axis

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig

/**
 * AXI4-Stream style bundle.
 * Source drives tvalid, tdata, tlast, tstrb, tid, tdest; sink drives tready.
 * Use Flipped(AxisBundle(...)) for sink side. Transfer occurs when tvalid && tready.
 *
 * Signals:
 *   tvalid  Source: this beat valid; tdata (and other payload) may be sampled.
 *   tready  Sink:   can accept this cycle; transfer completes when both high.
 *   tdata   Payload, width dataWidth.
 *   tlast   Packet end; high on final beat of a stream packet.
 *   tstrb   Byte strobe: 1 bit per TDATA byte; 1 = valid, 0 = null/placeholder.
 *   tid     Transaction ID: source/transaction identity for mux, reorder, or req–resp match.
 *   tdest   Destination ID: routing target when multiplexing.
 *
 * @param dataWidth  TDATA width (bits). tstrb length = (dataWidth/8).max(1).
 * @param idWidth    TID width; 0 = unused.
 * @param destWidth  TDEST width; 0 = unused.
 */
class AxisBundle(
  val b:           GlobalConfig,
  val dataWidth:   Int,
  val idWidth:     Int = 0,
  val destWidth:   Int = 0,
  val offsetWidth: Int = 0)
    extends Bundle {
  val tdata = UInt(dataWidth.W)
  val tlast = Bool()
  val tstrb = UInt((dataWidth / 8).max(1).W)
  val tid   = UInt(idWidth.W)
  val tdest = UInt(destWidth.W)

  val wmode  = Bool() // true=accumulator mode, false=direct write mode
  val offset = UInt(offsetWidth.W)
  val rob_id = UInt(log2Ceil(b.frontend.rob_entries).W)
}
