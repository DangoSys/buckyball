package framework.frontend.globalrs

import chisel3._
import chisel3.util._

// DPI-C BlackBox for instruction trace (issue / complete events)
class ITraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock       = Input(Clock())
    val reset       = Input(Bool())
    val is_issue    = Input(UInt(8.W))
    val rob_id      = Input(UInt(32.W))
    val domain_id   = Input(UInt(32.W))
    val funct       = Input(UInt(32.W))
    val pc          = Input(UInt(64.W))
    val rs1_idx     = Input(UInt(64.W))
    val rs2_idx     = Input(UInt(64.W))
    val rs1_data    = Input(UInt(64.W))
    val rs2_data    = Input(UInt(64.W))
    val bank_enable = Input(UInt(8.W))
    val enable      = Input(Bool())
  })

  setInline(
    "ITraceDPI.v",
    """
      |import "DPI-C" context function void dpi_itrace(
      |  input int unsigned is_issue,
      |  input int unsigned rob_id,
      |  input int unsigned domain_id,
      |  input int unsigned funct,
      |  input int unsigned pc_lo,
      |  input int unsigned pc_hi,
      |  input int unsigned rs1_idx_lo,
      |  input int unsigned rs1_idx_hi,
      |  input int unsigned rs2_idx_lo,
      |  input int unsigned rs2_idx_hi,
      |  input int unsigned rs1_data_lo,
      |  input int unsigned rs1_data_hi,
      |  input int unsigned rs2_data_lo,
      |  input int unsigned rs2_data_hi,
      |  input int unsigned bank_enable
      |);
      |
      |module ITraceDPI(
      |  input clock,
      |  input reset,
      |  input [7:0] is_issue,
      |  input [31:0] rob_id,
      |  input [31:0] domain_id,
      |  input [31:0] funct,
      |  input [63:0] pc,
      |  input [63:0] rs1_idx,
      |  input [63:0] rs2_idx,
      |  input [63:0] rs1_data,
      |  input [63:0] rs2_data,
      |  input [7:0] bank_enable,
      |  input enable
      |);
      |  reg [7:0]  is_issue_reg;
      |  reg [31:0] rob_id_reg;
      |  reg [31:0] domain_id_reg;
      |  reg [31:0] funct_reg;
      |  reg [63:0] pc_reg;
      |  reg [63:0] rs1_idx_reg;
      |  reg [63:0] rs2_idx_reg;
      |  reg [63:0] rs1_data_reg;
      |  reg [63:0] rs2_data_reg;
      |  reg [7:0]  bank_enable_reg;
      |  reg        valid_reg;
      |
      |  always @(posedge clock) begin
      |    if (reset) begin
      |      valid_reg <= 1'b0;
      |    end else begin
      |      if (valid_reg) begin
      |        dpi_itrace(is_issue_reg, rob_id_reg, domain_id_reg, funct_reg, pc_reg[31:0], pc_reg[63:32], rs1_idx_reg[31:0], rs1_idx_reg[63:32], rs2_idx_reg[31:0], rs2_idx_reg[63:32], rs1_data_reg[31:0], rs1_data_reg[63:32], rs2_data_reg[31:0], rs2_data_reg[63:32], bank_enable_reg);
      |      end
      |
      |      valid_reg <= enable;
      |      if (enable) begin
      |        is_issue_reg    <= is_issue;
      |        rob_id_reg      <= rob_id;
      |        domain_id_reg   <= domain_id;
      |        funct_reg       <= funct;
      |        pc_reg          <= pc;
      |        rs1_idx_reg     <= rs1_idx;
      |        rs2_idx_reg     <= rs2_idx;
      |        rs1_data_reg    <= rs1_data;
      |        rs2_data_reg    <= rs2_data;
      |        bank_enable_reg <= bank_enable;
      |      end
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
