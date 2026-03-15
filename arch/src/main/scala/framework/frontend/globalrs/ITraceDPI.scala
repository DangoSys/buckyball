package framework.frontend.globalrs

import chisel3._
import chisel3.util._

// DPI-C BlackBox for instruction trace (issue / complete events)
class ITraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val is_issue    = Input(UInt(8.W))
    val rob_id      = Input(UInt(32.W))
    val domain_id   = Input(UInt(32.W))
    val funct       = Input(UInt(32.W))
    val rs1         = Input(UInt(64.W))
    val rs2         = Input(UInt(64.W))
    val bank_enable = Input(UInt(8.W))
    val enable      = Input(Bool())
  })

  setInline(
    "ITraceDPI.v",
    """
      |import "DPI-C" function void dpi_itrace(
      |  input byte unsigned is_issue,
      |  input int unsigned rob_id,
      |  input int unsigned domain_id,
      |  input int unsigned funct,
      |  input longint unsigned rs1,
      |  input longint unsigned rs2,
      |  input byte unsigned bank_enable
      |);
      |
      |module ITraceDPI(
      |  input [7:0] is_issue,
      |  input [31:0] rob_id,
      |  input [31:0] domain_id,
      |  input [31:0] funct,
      |  input [63:0] rs1,
      |  input [63:0] rs2,
      |  input [7:0] bank_enable,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_itrace(is_issue, rob_id, domain_id, funct, rs1, rs2, bank_enable);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
