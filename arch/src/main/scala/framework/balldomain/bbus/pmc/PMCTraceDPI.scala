package framework.balldomain.bbus.pmc

import chisel3._
import chisel3.util._

class PMCTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock   = Input(Clock())
    val reset   = Input(Bool())
    val ball_id = Input(UInt(32.W))
    val rob_id  = Input(UInt(32.W))
    val elapsed = Input(UInt(64.W))
    val enable  = Input(Bool())
  })

  setInline(
    "PMCTraceDPI.v",
    """
      |import "DPI-C" context function void dpi_pmctrace(
      |  input int unsigned ball_id,
      |  input int unsigned rob_id,
      |  input int unsigned elapsed_lo,
      |  input int unsigned elapsed_hi
      |);
      |
      |module PMCTraceDPI(
      |  input clock,
      |  input reset,
      |  input [31:0] ball_id,
      |  input [31:0] rob_id,
      |  input [63:0] elapsed,
      |  input enable
      |);
      |  reg [31:0] ball_id_reg;
      |  reg [31:0] rob_id_reg;
      |  reg [63:0] elapsed_reg;
      |  reg        valid_reg;
      |
      |  always @(posedge clock) begin
      |    if (reset) begin
      |      valid_reg <= 1'b0;
      |    end else begin
      |      if (valid_reg) begin
      |        dpi_pmctrace(ball_id_reg, rob_id_reg, elapsed_reg[31:0], elapsed_reg[63:32]);
      |      end
      |
      |      valid_reg <= enable;
      |      if (enable) begin
      |        ball_id_reg <= ball_id;
      |        rob_id_reg  <= rob_id;
      |        elapsed_reg <= elapsed;
      |      end
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
