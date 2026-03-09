package framework.memdomain.utils.pmc

import chisel3._
import chisel3.util._

// DPI-C BlackBox for memory PMC trace
class MemPMCTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val is_store = Input(UInt(8.W))
    val rob_id   = Input(UInt(32.W))
    val elapsed  = Input(UInt(64.W))
    val enable   = Input(Bool())
  })

  setInline(
    "MemPMCTraceDPI.v",
    """
      |import "DPI-C" function void dpi_mem_pmctrace(
      |  input byte unsigned is_store,
      |  input int unsigned rob_id,
      |  input longint unsigned elapsed
      |);
      |
      |module MemPMCTraceDPI(
      |  input [7:0] is_store,
      |  input [31:0] rob_id,
      |  input [63:0] elapsed,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_mem_pmctrace(is_store, rob_id, elapsed);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
