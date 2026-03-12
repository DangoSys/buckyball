package framework.balldomain.prototype.trace

import chisel3._
import chisel3.util._

/**
 * DPI-C BlackBox for cycle counter trace.
 * Outputs [CTRACE] lines to bdb.log.
 */
class CTraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val subcmd  = Input(UInt(8.W))
    val ctr_id  = Input(UInt(32.W))
    val tag     = Input(UInt(64.W))
    val elapsed = Input(UInt(64.W))
    val cycle   = Input(UInt(64.W))
    val enable  = Input(Bool())
  })

  setInline(
    "CTraceDPI.v",
    """
      |import "DPI-C" function void dpi_ctrace(
      |  input byte unsigned subcmd,
      |  input int unsigned ctr_id,
      |  input longint unsigned tag,
      |  input longint unsigned elapsed,
      |  input longint unsigned cycle
      |);
      |
      |module CTraceDPI(
      |  input [7:0]  subcmd,
      |  input [31:0] ctr_id,
      |  input [63:0] tag,
      |  input [63:0] elapsed,
      |  input [63:0] cycle,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_ctrace(subcmd, ctr_id, tag, elapsed, cycle);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}

/**
 * DPI-C BlackBox for backdoor get_read_addr.
 * Returns packed [63:32]=bank_id, [31:0]=row.
 */
class BackdoorGetReadAddrDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val result = Output(UInt(64.W))
    val enable = Input(Bool())
  })

  setInline(
    "BackdoorGetReadAddrDPI.v",
    """
      |import "DPI-C" function longint unsigned dpi_backdoor_get_read_addr();
      |
      |module BackdoorGetReadAddrDPI(
      |  output [63:0] result,
      |  input enable
      |);
      |  reg [63:0] result_reg;
      |  assign result = result_reg;
      |  always @(*) begin
      |    result_reg = 64'd0;
      |    if (enable) begin
      |      result_reg = dpi_backdoor_get_read_addr();
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}

/**
 * DPI-C BlackBox for backdoor get_write_addr.
 * Returns packed [63:32]=bank_id, [31:0]=row.
 */
class BackdoorGetWriteAddrDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val result = Output(UInt(64.W))
    val enable = Input(Bool())
  })

  setInline(
    "BackdoorGetWriteAddrDPI.v",
    """
      |import "DPI-C" function longint unsigned dpi_backdoor_get_write_addr();
      |
      |module BackdoorGetWriteAddrDPI(
      |  output [63:0] result,
      |  input enable
      |);
      |  reg [63:0] result_reg;
      |  assign result = result_reg;
      |  always @(*) begin
      |    result_reg = 64'd0;
      |    if (enable) begin
      |      result_reg = dpi_backdoor_get_write_addr();
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}

/**
 * DPI-C BlackBox for backdoor get_write_data.
 * Returns 128-bit data as two 64-bit outputs.
 */
class BackdoorGetWriteDataDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val data_lo = Output(UInt(64.W))
    val data_hi = Output(UInt(64.W))
    val enable  = Input(Bool())
  })

  setInline(
    "BackdoorGetWriteDataDPI.v",
    """
      |import "DPI-C" function void dpi_backdoor_get_write_data(
      |  output longint unsigned data_lo,
      |  output longint unsigned data_hi
      |);
      |
      |module BackdoorGetWriteDataDPI(
      |  output [63:0] data_lo,
      |  output [63:0] data_hi,
      |  input enable
      |);
      |  reg [63:0] data_lo_reg;
      |  reg [63:0] data_hi_reg;
      |  assign data_lo = data_lo_reg;
      |  assign data_hi = data_hi_reg;
      |  always @(*) begin
      |    data_lo_reg = 64'd0;
      |    data_hi_reg = 64'd0;
      |    if (enable) begin
      |      dpi_backdoor_get_write_data(data_lo_reg, data_hi_reg);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}

/**
 * DPI-C BlackBox for backdoor put_read_data.
 * Reports read data back to C++ for logging.
 */
class BackdoorPutReadDataDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val bank_id = Input(UInt(32.W))
    val row     = Input(UInt(32.W))
    val data_lo = Input(UInt(64.W))
    val data_hi = Input(UInt(64.W))
    val enable  = Input(Bool())
  })

  setInline(
    "BackdoorPutReadDataDPI.v",
    """
      |import "DPI-C" function void dpi_backdoor_put_read_data(
      |  input int unsigned bank_id,
      |  input int unsigned row,
      |  input longint unsigned data_lo,
      |  input longint unsigned data_hi
      |);
      |
      |module BackdoorPutReadDataDPI(
      |  input [31:0] bank_id,
      |  input [31:0] row,
      |  input [63:0] data_lo,
      |  input [63:0] data_hi,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_backdoor_put_read_data(bank_id, row, data_lo, data_hi);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}

/**
 * DPI-C BlackBox for backdoor put_write_done.
 * Reports write completion back to C++ for logging.
 */
class BackdoorPutWriteDoneDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val bank_id = Input(UInt(32.W))
    val row     = Input(UInt(32.W))
    val data_lo = Input(UInt(64.W))
    val data_hi = Input(UInt(64.W))
    val enable  = Input(Bool())
  })

  setInline(
    "BackdoorPutWriteDoneDPI.v",
    """
      |import "DPI-C" function void dpi_backdoor_put_write_done(
      |  input int unsigned bank_id,
      |  input int unsigned row,
      |  input longint unsigned data_lo,
      |  input longint unsigned data_hi
      |);
      |
      |module BackdoorPutWriteDoneDPI(
      |  input [31:0] bank_id,
      |  input [31:0] row,
      |  input [63:0] data_lo,
      |  input [63:0] data_hi,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_backdoor_put_write_done(bank_id, row, data_lo, data_hi);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}
