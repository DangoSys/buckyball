package framework.memdomain.backend

import chisel3._
import chisel3.util._

// DPI-C BlackBox for memory trace
class MTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock     = Input(Clock())
    val reset     = Input(Bool())
    val is_write  = Input(UInt(8.W))
    val is_shared = Input(UInt(8.W))
    val channel   = Input(UInt(32.W))
    val hart_id   = Input(UInt(64.W))
    val vbank_id  = Input(UInt(32.W))
    val pbank_id  = Input(UInt(32.W))
    val group_id  = Input(UInt(32.W))
    val addr      = Input(UInt(32.W))
    val data_lo   = Input(UInt(64.W))
    val data_hi   = Input(UInt(64.W))
    val enable    = Input(Bool())
  })

  setInline(
    "MTraceDPI.v",
    """
      |import "DPI-C" context function void dpi_mtrace(
      |  input int unsigned is_write,
      |  input int unsigned is_shared,
      |  input int unsigned channel,
      |  input int unsigned hart_id_lo,
      |  input int unsigned hart_id_hi,
      |  input int unsigned vbank_id,
      |  input int unsigned pbank_id,
      |  input int unsigned group_id,
      |  input int unsigned addr,
      |  input int unsigned data_lo_lo,
      |  input int unsigned data_lo_hi,
      |  input int unsigned data_hi_lo,
      |  input int unsigned data_hi_hi
      |);
      |
      |module MTraceDPI(
      |  input clock,
      |  input reset,
      |  input [7:0] is_write,
      |  input [7:0] is_shared,
      |  input [31:0] channel,
      |  input [63:0] hart_id,
      |  input [31:0] vbank_id,
      |  input [31:0] pbank_id,
      |  input [31:0] group_id,
      |  input [31:0] addr,
      |  input [63:0] data_lo,
      |  input [63:0] data_hi,
      |  input enable
      |);
      |  reg [7:0]  is_write_reg;
      |  reg [7:0]  is_shared_reg;
      |  reg [31:0] channel_reg;
      |  reg [63:0] hart_id_reg;
      |  reg [31:0] vbank_id_reg;
      |  reg [31:0] pbank_id_reg;
      |  reg [31:0] group_id_reg;
      |  reg [31:0] addr_reg;
      |  reg [63:0] data_lo_reg;
      |  reg [63:0] data_hi_reg;
      |  reg        valid_reg;
      |
      |  always @(posedge clock) begin
      |    if (reset) begin
      |      valid_reg <= 1'b0;
      |    end else begin
      |      if (valid_reg) begin
      |        dpi_mtrace(is_write_reg, is_shared_reg, channel_reg, hart_id_reg[31:0], hart_id_reg[63:32], vbank_id_reg, pbank_id_reg, group_id_reg, addr_reg, data_lo_reg[31:0], data_lo_reg[63:32], data_hi_reg[31:0], data_hi_reg[63:32]);
      |      end
      |
      |      valid_reg <= enable;
      |      if (enable) begin
      |        is_write_reg  <= is_write;
      |        is_shared_reg <= is_shared;
      |        channel_reg   <= channel;
      |        hart_id_reg   <= hart_id;
      |        vbank_id_reg  <= vbank_id;
      |        pbank_id_reg  <= pbank_id;
      |        group_id_reg  <= group_id;
      |        addr_reg      <= addr;
      |        data_lo_reg   <= data_lo;
      |        data_hi_reg   <= data_hi;
      |      end
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
