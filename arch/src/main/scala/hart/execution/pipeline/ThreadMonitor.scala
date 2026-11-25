package framework.hart.execution.pipeline

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.dpi.DpiGuard

/**
 * The commands a hart's CPU gives its controller (the threads' tasks, loads and stores) and their
 * responses, one event a cycle each, and every 65536 cycles the CPU's pc and sequencer state, under
 * itrace: a command is its pc, funct7 and operands; a response its destination and data.
 */
@instantiable
class ThreadMonitor extends BlackBox with HasBlackBoxInline {

  @public
  val io = IO(new Bundle {
    val clock    = Input(Clock())
    val reset    = Input(Bool())
    val command  = Input(Bool())
    val pc       = Input(UInt(64.W))
    val funct    = Input(UInt(7.W))
    val rs1      = Input(UInt(64.W))
    val rs2      = Input(UInt(64.W))
    val response = Input(Bool())
    val rd       = Input(UInt(5.W))
    val data     = Input(UInt(64.W))
    val state    = Input(UInt(8.W))
  })

  setInline(
    "ThreadMonitor.v",
    """
      |module ThreadMonitor(
      |  input clock, input reset,
      |  input command, input [63:0] pc, input [6:0] funct, input [63:0] rs1, input [63:0] rs2,
      |  input response, input [4:0] rd, input [63:0] data, input [7:0] state
      |);
      |""".stripMargin + DpiGuard.wrapITrace(
      """
        |  import "DPI-C" context function void dpi_thread_monitor(
        |    input int unsigned kind, input int unsigned code,
        |    input int unsigned a_lo, input int unsigned a_hi,
        |    input int unsigned b_lo, input int unsigned b_hi,
        |    input int unsigned c_lo, input int unsigned c_hi
        |  );
        |  reg [15:0] beat = 16'd0;
        |  always @(posedge clock) begin
        |    beat <= beat + 16'd1;
        |    if (!reset && command)
        |      dpi_thread_monitor(0, {25'b0, funct}, pc[31:0], pc[63:32], rs1[31:0], rs1[63:32], rs2[31:0], rs2[63:32]);
        |    if (!reset && response)
        |      dpi_thread_monitor(1, {27'b0, rd}, data[31:0], data[63:32], 0, 0, 0, 0);
        |    if (!reset && beat == 16'd0)
        |      dpi_thread_monitor(2, {24'b0, state}, pc[31:0], pc[63:32], 0, 0, 0, 0);
        |  end
        |""".stripMargin
    ) + "endmodule\n"
  )
}
