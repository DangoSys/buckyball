package framework.memdomain.backend

import chisel3._
import chisel3.util._

// DPI-C BlackBox for memory trace
class MTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val is_write = Input(UInt(8.W))
    val channel  = Input(UInt(32.W))
    val vbank_id = Input(UInt(32.W))
    val group_id = Input(UInt(32.W))
    val addr     = Input(UInt(32.W))
    val data_lo  = Input(UInt(64.W))
    val data_hi  = Input(UInt(64.W))
    val enable   = Input(Bool())
  })

  setInline(
    "MTraceDPI.v",
    """
      |import "DPI-C" function void dpi_mtrace(
      |  input byte unsigned is_write,
      |  input int unsigned channel,
      |  input int unsigned vbank_id,
      |  input int unsigned group_id,
      |  input int unsigned addr,
      |  input longint unsigned data_lo,
      |  input longint unsigned data_hi
      |);
      |
      |module MTraceDPI(
      |  input [7:0] is_write,
      |  input [31:0] channel,
      |  input [31:0] vbank_id,
      |  input [31:0] group_id,
      |  input [31:0] addr,
      |  input [63:0] data_lo,
      |  input [63:0] data_hi,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_mtrace(is_write, channel, vbank_id, group_id, addr, data_lo, data_hi);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
