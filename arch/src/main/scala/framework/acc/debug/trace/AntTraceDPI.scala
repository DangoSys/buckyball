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
        |  always @(posedge clock) begin
        |    if (!reset && enable)
        |      dpi_ant_trace(thread_id[31:0], thread_id[63:32], pc[31:0], pc[63:32],
        |                    instruction, {27'b0, rd}, data[31:0], data[63:32]);
        |  end
        |""".stripMargin
    ) + "endmodule\n"
  )
}
