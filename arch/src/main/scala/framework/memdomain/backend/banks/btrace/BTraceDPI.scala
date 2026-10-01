package framework.memdomain.backend.banks.btrace

import chisel3._
import chisel3.util._
import framework.dpi.DpiGuard

class BTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock    = Input(Clock())
    val reset    = Input(Bool())
    val instId   = Input(UInt(64.W))
    val hartId   = Input(UInt(64.W))
    val w0Vbank  = Input(UInt(32.W))
    val w0Hash   = Input(UInt(32.W))
    val fire     = Input(Bool())
    val produced = Input(UInt(64.W))
    val idle     = Input(Bool())
  })

  setInline(
    "BTraceDPI.v",
    """
      |module BTraceDPI(
      |  input clock,
      |  input reset,
      |  input [63:0] instId,
      |  input [63:0] hartId,
      |  input [31:0] w0Vbank,
      |  input [31:0] w0Hash,
      |  input fire,
      |  input [63:0] produced,
      |  input idle
      |);
      |""".stripMargin + DpiGuard.wrapBTrace(
      """
        |  export "DPI-C" function btrace_snapshot;
        |  function void btrace_snapshot(
        |    output int unsigned hart_lo,
        |    output int unsigned hart_hi,
        |    output int unsigned produced_lo,
        |    output int unsigned produced_hi,
        |    output int unsigned is_idle
        |  );
        |    hart_lo = hartId[31:0];
        |    hart_hi = hartId[63:32];
        |    produced_lo = produced[31:0];
        |    produced_hi = produced[63:32];
        |    is_idle = {31'b0, idle};
        |  endfunction
        |  import "DPI-C" context function void dpi_btrace(
        |    input int unsigned inst_id_lo,
        |    input int unsigned inst_id_hi,
        |    input int unsigned hart_id_lo,
        |    input int unsigned hart_id_hi,
        |    input int unsigned w0_vbank,
        |    input int unsigned w0_hash
        |  );
        |  always @(posedge clock) begin
        |    if (!reset && fire) begin
        |      dpi_btrace(instId[31:0], instId[63:32], hartId[31:0], hartId[63:32], w0Vbank, w0Hash);
        |    end
        |  end
        |""".stripMargin
    ) +
      """
        |endmodule
        |""".stripMargin
  )
}
