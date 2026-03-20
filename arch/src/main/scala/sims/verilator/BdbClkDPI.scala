package sims.verilator

import chisel3._
import chisel3.util.HasBlackBoxInline

/**
 * Pushes harness reference clock cycle index into C++ via DPI each posedge.
 * Matches BBSimHarness.clock (see ball_exec_once: clock=1 half-cycle).
 */
class BdbClkDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
  })

  setInline(
    "BdbClkDPI.v",
    """
      |import "DPI-C" function void dpi_bdb_set_clk(input longint unsigned c);
      |module BdbClkDPI(
      |  input wire clock,
      |  input wire reset
      |);
      |  reg [63:0] cnt;
      |  always @(posedge clock) begin
      |    if (reset) begin
      |      cnt <= 64'd0;
      |      dpi_bdb_set_clk(64'd0);
      |    end else begin
      |      dpi_bdb_set_clk(cnt);
      |      cnt <= cnt + 64'd1;
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
