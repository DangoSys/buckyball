package framework.balldomain.prototype.trace

import chisel3._
import chisel3.util._

/**
 * DPI-C BlackBox for cycle counter trace.
 * Outputs [CTRACE] lines to bdb.log.
 */
class CTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val subcmd  = Input(UInt(8.W))
    val ctr_id  = Input(UInt(32.W))
    val tag     = Input(UInt(64.W))
    val elapsed = Input(UInt(64.W))
    val cycle   = Input(UInt(64.W))
    val enable  = Input(Bool())
  })

  setInline(
    "CTraceDPI.v",
    """
      |import "DPI-C" function void dpi_ctrace(
      |  input byte unsigned subcmd,
      |  input int unsigned ctr_id,
      |  input longint unsigned tag,
      |  input longint unsigned elapsed,
      |  input longint unsigned cycle
      |);
      |
      |module CTraceDPI(
      |  input [7:0]  subcmd,
      |  input [31:0] ctr_id,
      |  input [63:0] tag,
      |  input [63:0] elapsed,
      |  input [63:0] cycle,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_ctrace(subcmd, ctr_id, tag, elapsed, cycle);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
