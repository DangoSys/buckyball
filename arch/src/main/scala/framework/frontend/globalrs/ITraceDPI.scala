package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import framework.dpi.DpiGuard

class ITraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock     = Input(Clock())
    val reset     = Input(Bool())
    val is_issue  = Input(UInt(8.W))
    val hart_id   = Input(UInt(64.W))
    val rob_id    = Input(UInt(32.W))
    val inst_id   = Input(UInt(64.W))
    val domain_id = Input(UInt(32.W))
    val funct     = Input(UInt(32.W))
    val pc        = Input(UInt(64.W))
    val rs1_data  = Input(UInt(64.W))
    val rs2_data  = Input(UInt(64.W))
    val enable    = Input(Bool())
  })

  setInline(
    "ITraceDPI.v",
    """
      |module ITraceDPI(
      |  input clock,
      |  input reset,
      |  input [7:0] is_issue,
      |  input [63:0] hart_id,
      |  input [31:0] rob_id,
      |  input [63:0] inst_id,
      |  input [31:0] domain_id,
      |  input [31:0] funct,
      |  input [63:0] pc,
      |  input [63:0] rs1_data,
      |  input [63:0] rs2_data,
      |  input enable
      |);
      |""".stripMargin + DpiGuard.wrapITrace("""
                                               |  import "DPI-C" context function void dpi_itrace(
                                               |    input int unsigned is_issue,
                                               |    input int unsigned hart_id_lo,
                                               |    input int unsigned hart_id_hi,
                                               |    input int unsigned rob_id,
                                               |    input int unsigned inst_id_lo,
                                               |    input int unsigned inst_id_hi,
                                               |    input int unsigned domain_id,
                                               |    input int unsigned funct,
                                               |    input int unsigned pc_lo,
                                               |    input int unsigned pc_hi,
                                               |    input int unsigned rs1_data_lo,
                                               |    input int unsigned rs1_data_hi,
                                               |    input int unsigned rs2_data_lo,
                                               |    input int unsigned rs2_data_hi
                                               |  );
                                               |  reg [7:0]  is_issue_reg;
                                               |  reg [63:0] hart_id_reg;
                                               |  reg [31:0] rob_id_reg;
                                               |  reg [63:0] inst_id_reg;
                                               |  reg [31:0] domain_id_reg;
                                               |  reg [31:0] funct_reg;
                                               |  reg [63:0] pc_reg;
                                               |  reg [63:0] rs1_data_reg;
                                               |  reg [63:0] rs2_data_reg;
                                               |  reg        valid_reg;
                                               |
                                               |  always @(posedge clock) begin
                                               |    if (reset) begin
                                               |      valid_reg <= 1'b0;
                                               |    end else begin
                                               |      if (valid_reg) begin
                                               |        dpi_itrace(is_issue_reg, hart_id_reg[31:0], hart_id_reg[63:32], rob_id_reg, inst_id_reg[31:0], inst_id_reg[63:32], domain_id_reg, funct_reg, pc_reg[31:0], pc_reg[63:32], rs1_data_reg[31:0], rs1_data_reg[63:32], rs2_data_reg[31:0], rs2_data_reg[63:32]);
                                               |      end
                                               |
                                               |      valid_reg <= enable;
                                               |      if (enable) begin
                                               |        is_issue_reg    <= is_issue;
                                               |        hart_id_reg     <= hart_id;
                                               |        rob_id_reg      <= rob_id;
                                               |        inst_id_reg     <= inst_id;
                                               |        domain_id_reg   <= domain_id;
                                               |        funct_reg       <= funct;
                                               |        pc_reg          <= pc;
                                               |        rs1_data_reg    <= rs1_data;
                                               |        rs2_data_reg    <= rs2_data;
                                               |      end
                                               |    end
                                               |  end
                                               |""".stripMargin) +
      """
        |endmodule
    """.stripMargin
  )
}
