package framework.balldomain.bbus.pmc

import chisel3._
import chisel3.util._

// DPI-C BlackBox for PMC trace
class PMCTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val ball_id = Input(UInt(32.W))
    val rob_id  = Input(UInt(32.W))
    val elapsed = Input(UInt(64.W))
    val enable  = Input(Bool())
  })

  setInline(
    "PMCTraceDPI.v",
    """
      |import "DPI-C" function void dpi_pmctrace(
      |  input int unsigned ball_id,
      |  input int unsigned rob_id,
      |  input longint unsigned elapsed
      |);
      |
      |module PMCTraceDPI(
      |  input [31:0] ball_id,
      |  input [31:0] rob_id,
      |  input [63:0] elapsed,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_pmctrace(ball_id, rob_id, elapsed);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
