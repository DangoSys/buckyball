package framework.acc.debug

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.lib.dpi.DpiGuard

@instantiable
class BankCheckDPI extends BlackBox with HasBlackBoxInline {

  @public
  val io = IO(new Bundle {
    val clock         = Input(Clock())
    val reset         = Input(Bool())
    val instId        = Input(UInt(64.W))
    val threadId      = Input(UInt(64.W))
    val ownerThreadId = Input(UInt(64.W))
    val w0Vbank       = Input(UInt(32.W))
    val w0Hash        = Input(UInt(32.W))
    val fire          = Input(Bool())
    val produced      = Input(UInt(64.W))
    val idle          = Input(Bool())
  })

  setInline(
    "BankCheckDPI.v",
    """
      |module BankCheckDPI(
      |  input clock,
      |  input reset,
      |  input [63:0] instId,
      |  input [63:0] threadId,
      |  input [63:0] ownerThreadId,
      |  input [31:0] w0Vbank,
      |  input [31:0] w0Hash,
      |  input fire,
      |  input [63:0] produced,
      |  input idle
      |);
      |""".stripMargin + DpiGuard.wrapBTrace(
      """
        |  reg [63:0] inst_id_reg;
        |  reg [63:0] thread_id_reg;
        |  reg [63:0] owner_thread_id_reg;
        |  reg [31:0] w0_vbank_reg;
        |  reg [31:0] w0_hash_reg;
        |  reg        valid_reg = 1'b0;
        |
        |  export "DPI-C" function btrace_snapshot;
        |  function void btrace_snapshot(
        |    output int unsigned thread_lo,
        |    output int unsigned thread_hi,
        |    output int unsigned produced_lo,
        |    output int unsigned produced_hi,
        |    output int unsigned is_idle
        |  );
        |    thread_lo = threadId[31:0];
        |    thread_hi = threadId[63:32];
        |    produced_lo = produced[31:0];
        |    produced_hi = produced[63:32];
        |    is_idle = {31'b0, idle && !valid_reg};
        |  endfunction
        |  import "DPI-C" context function void dpi_btrace(
        |    input int unsigned inst_id_lo,
        |    input int unsigned inst_id_hi,
        |    input int unsigned thread_id_lo,
        |    input int unsigned thread_id_hi,
        |    input int unsigned owner_thread_id_lo,
        |    input int unsigned owner_thread_id_hi,
        |    input int unsigned w0_vbank,
        |    input int unsigned w0_hash
        |  );
        |  always @(posedge clock) begin
        |    // Drain the sampled event even when reset starts on this edge.
        |    if (valid_reg) begin
        |      dpi_btrace(inst_id_reg[31:0], inst_id_reg[63:32], thread_id_reg[31:0], thread_id_reg[63:32], owner_thread_id_reg[31:0], owner_thread_id_reg[63:32], w0_vbank_reg, w0_hash_reg);
        |    end
        |
        |    valid_reg <= !reset && fire;
        |    if (!reset && fire) begin
        |      inst_id_reg         <= instId;
        |      thread_id_reg       <= threadId;
        |      owner_thread_id_reg <= ownerThreadId;
        |      w0_vbank_reg        <= w0Vbank;
        |      w0_hash_reg         <= w0Hash;
        |    end
        |  end
        |""".stripMargin
    ) +
      """
        |endmodule
        |""".stripMargin
  )
}
