package examples.balls.trace

import chisel3._
import chisel3.util._

/**
 * DPI-C BlackBox for cycle counter trace.
 * Outputs [CTRACE] lines to bdb.log.
 */
class CTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock   = Input(Clock())
    val reset   = Input(Bool())
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
      |import "DPI-C" context function void dpi_ctrace(
      |  input int unsigned subcmd,
      |  input int unsigned ctr_id,
      |  input int unsigned tag_lo,
      |  input int unsigned tag_hi,
      |  input int unsigned elapsed_lo,
      |  input int unsigned elapsed_hi,
      |  input int unsigned cycle_lo,
      |  input int unsigned cycle_hi
      |);
      |
      |module CTraceDPI(
      |  input clock,
      |  input reset,
      |  input [7:0]  subcmd,
      |  input [31:0] ctr_id,
      |  input [63:0] tag,
      |  input [63:0] elapsed,
      |  input [63:0] cycle,
      |  input enable
      |);
      |  reg [7:0]  subcmd_reg;
      |  reg [31:0] ctr_id_reg;
      |  reg [63:0] tag_reg;
      |  reg [63:0] elapsed_reg;
      |  reg [63:0] cycle_reg;
      |  reg        valid_reg;
      |
      |  always @(posedge clock) begin
      |    if (reset) begin
      |      valid_reg <= 1'b0;
      |    end else begin
      |      if (valid_reg) begin
      |        dpi_ctrace(subcmd_reg, ctr_id_reg, tag_reg[31:0], tag_reg[63:32], elapsed_reg[31:0], elapsed_reg[63:32], cycle_reg[31:0], cycle_reg[63:32]);
      |      end
      |
      |      valid_reg <= enable;
      |      if (enable) begin
      |        subcmd_reg  <= subcmd;
      |        ctr_id_reg  <= ctr_id;
      |        tag_reg     <= tag;
      |        elapsed_reg <= elapsed;
      |        cycle_reg   <= cycle;
      |      end
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
