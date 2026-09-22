package framework.memdomain.backend.banks.btrace

import chisel3._
import chisel3.util._
import framework.dpi.DpiGuard

class BTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock   = Input(Clock())
    val reset   = Input(Bool())
    val instId  = Input(UInt(64.W))
    val hartId  = Input(UInt(64.W))
    val r0Vbank = Input(UInt(32.W))
    val r0Hash  = Input(UInt(32.W))
    val r1Vbank = Input(UInt(32.W))
    val r1Hash  = Input(UInt(32.W))
    val w0Vbank = Input(UInt(32.W))
    val w0Hash  = Input(UInt(32.W))
    val fire    = Input(Bool())
  })

  setInline(
    "BTraceDPI.v",
    """
      |module BTraceDPI(
      |  input clock,
      |  input reset,
      |  input [63:0] instId,
      |  input [63:0] hartId,
      |  input [31:0] r0Vbank,
      |  input [31:0] r0Hash,
      |  input [31:0] r1Vbank,
      |  input [31:0] r1Hash,
      |  input [31:0] w0Vbank,
      |  input [31:0] w0Hash,
      |  input fire
      |);
      |""".stripMargin + DpiGuard.wrapBTrace(
      """
        |  import "DPI-C" context function void dpi_btrace(
        |    input int unsigned inst_id_lo,
        |    input int unsigned inst_id_hi,
        |    input int unsigned hart_id_lo,
        |    input int unsigned hart_id_hi,
        |    input int unsigned r0_vbank,
        |    input int unsigned r0_hash,
        |    input int unsigned r1_vbank,
        |    input int unsigned r1_hash,
        |    input int unsigned w0_vbank,
        |    input int unsigned w0_hash
        |  );
        |  always @(posedge clock) begin
        |    if (!reset && fire) begin
        |      dpi_btrace(instId[31:0], instId[63:32], hartId[31:0], hartId[63:32], r0Vbank, r0Hash, r1Vbank, r1Hash, w0Vbank, w0Hash);
        |    end
        |  end
        |""".stripMargin
    ) +
      """
        |endmodule
        |""".stripMargin
  )
}
