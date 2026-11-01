package framework.acc.debug.trace

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.lib.dpi.DpiGuard

@instantiable
class AntTraceDPI extends BlackBox with HasBlackBoxInline {

  @public
  val io = IO(new Bundle {
    val clock       = Input(Clock())
    val reset       = Input(Bool())
    val thread_id   = Input(UInt(64.W))
    val pc          = Input(UInt(64.W))
    val instruction = Input(UInt(32.W))
    val rd          = Input(UInt(5.W))
    val data        = Input(UInt(64.W))
    val enable      = Input(Bool())
  })

  setInline(
    "AntTraceDPI.v",
    """
      |module AntTraceDPI(
      |  input clock, input reset, input enable,
      |  input [63:0] thread_id, input [63:0] pc,
      |  input [31:0] instruction, input [4:0] rd, input [63:0] data
      |);
      |""".stripMargin + DpiGuard.wrapITrace(
      """
        |  import "DPI-C" context function void dpi_ant_trace(
        |    input int unsigned thread_lo, input int unsigned thread_hi,
        |    input int unsigned pc_lo, input int unsigned pc_hi,
        |    input int unsigned instruction, input int unsigned rd,
        |    input int unsigned data_lo, input int unsigned data_hi
        |  );
        |  reg [63:0] thread_id_reg;
        |  reg [63:0] pc_reg;
        |  reg [31:0] instruction_reg;
        |  reg [4:0]  rd_reg;
        |  reg [63:0] data_reg;
        |  reg        valid_reg = 1'b0;
        |
        |  always @(posedge clock) begin
        |    if (valid_reg) begin
        |      dpi_ant_trace(thread_id_reg[31:0], thread_id_reg[63:32], pc_reg[31:0], pc_reg[63:32],
        |                    instruction_reg, {27'b0, rd_reg}, data_reg[31:0], data_reg[63:32]);
        |    end
        |
        |    valid_reg <= !reset && enable;
        |    if (!reset && enable) begin
        |      thread_id_reg   <= thread_id;
        |      pc_reg          <= pc;
        |      instruction_reg <= instruction;
        |      rd_reg          <= rd;
        |      data_reg        <= data;
        |    end
        |  end
        |""".stripMargin
    ) + "endmodule\n"
  )
}
