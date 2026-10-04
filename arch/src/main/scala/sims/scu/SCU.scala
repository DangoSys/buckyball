package sims.scu

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util.HasBlackBoxInline

/**
 * SCU is a global multi-hart System Control Unit (see SystemControl). It
 * provides a per-hart sub-region for UART output and simulation exit. Each hart accesses its own sub-region via
 * `baseAddress + hartId * strideBytes`.
 *
 * Address layout (default values):
 *   baseAddress     = 0x60000000
 *   strideBytes     = 0x40000   (256 KiB per hart)
 *   totalSizeBytes  = 0x10000000 (256 MiB total, supports up to 1024 harts)
 *   maxHarts        = 64
 *
 * Within each hart's sub-region:
 *   +0x00000: simExit register (write triggers $finish with exit code)
 *   +0x20000: uartTx register  (write outputs a byte via DPI-C)
 *   +0x20004: uartRx register  (read pops a byte from host input)
 *   +0x20005: status register  (read-only, bit0 rx-valid, bit5/6 tx-ready)
 *   +0x20006: ready register   (read/write software-visible byte)
 *
 * @param baseAddress   physical base address (must be aligned to totalSizeBytes)
 * @param strideBytes   bytes per hart (must be power of two)
 * @param totalSizeBytes total address window (must be power of two,
 *                       >= maxHarts * strideBytes)
 * @param maxHarts      number of hart sub-regions. Accesses targeting
 *                      hartId >= maxHarts return an access error.
 */
case class SCUParams(
  baseAddress:    BigInt = BigInt("60000000", 16),
  strideBytes:    BigInt = BigInt("40000", 16),
  totalSizeBytes: BigInt = BigInt("10000000", 16),
  maxHarts:       Int = 64) {
  require(
    strideBytes > 0 && (strideBytes & (strideBytes - 1)) == 0,
    s"SCU strideBytes ($strideBytes) must be a positive power of two"
  )
  require(
    totalSizeBytes > 0 && (totalSizeBytes & (totalSizeBytes - 1)) == 0,
    s"SCU totalSizeBytes ($totalSizeBytes) must be a positive power of two"
  )
  require(maxHarts > 0, s"SCU maxHarts ($maxHarts) must be positive")
  require(
    BigInt(maxHarts) * strideBytes <= totalSizeBytes,
    s"SCU maxHarts ($maxHarts) * strideBytes ($strideBytes) = " +
      s"${BigInt(maxHarts) * strideBytes} exceeds totalSizeBytes ($totalSizeBytes)"
  )
}

/**
 * Single DPI-C bridge module for all harts. The hart_id is supplied as an
 * input signal rather than being baked into the module name, so only one
 * BlackBox is generated and DPI-C imports are not duplicated.
 *
 * Has separate hart_id inputs for uart and exit operations because different
 * harts may write uart and exit simultaneously.
 */
@instantiable
class SCUWriteDPI extends BlackBox with HasBlackBoxInline {
  override def desiredName = "SCUWriteDPI"

  @public val io = IO(new Bundle {
    val clock        = Input(Clock())
    val reset        = Input(Bool())
    val uart_hart_id = Input(UInt(32.W))
    val uart_valid   = Input(Bool())
    val uart_data    = Input(UInt(8.W))
    val exit_hart_id = Input(UInt(32.W))
    val exit_valid   = Input(Bool())
    val exit_code    = Input(UInt(32.W))
  })

  setInline(
    "SCUWriteDPI.v",
    s"""
       |import "DPI-C" context function void scu_uart_write(input int unsigned hart_id, input int unsigned ch);
       |import "DPI-C" context function void scu_sim_exit(input int unsigned hart_id, input int unsigned code);
       |
       |module SCUWriteDPI(
       |  input         clock,
       |  input         reset,
       |  input  [31:0] uart_hart_id,
       |  input         uart_valid,
       |  input  [7:0]  uart_data,
       |  input  [31:0] exit_hart_id,
       |  input         exit_valid,
       |  input  [31:0] exit_code
       |);
       |  always @(posedge clock) begin
       |    if (!reset) begin
       |      if (uart_valid) begin
       |        scu_uart_write(uart_hart_id, {24'h0, uart_data});
       |      end
       |
       |      if (exit_valid) begin
       |        scu_sim_exit(exit_hart_id, exit_code);
       |      end
       |    end
       |  end
       |endmodule
    """.stripMargin
  )
}

@instantiable
class SCUReadDPI extends BlackBox with HasBlackBoxInline {
  override def desiredName = "SCUReadDPI"

  @public val io = IO(new Bundle {
    val clock    = Input(Clock())
    val reset    = Input(Bool())
    val hart_id  = Input(UInt(32.W))
    val enable   = Input(Bool())
    val pop      = Input(Bool())
    val rx_valid = Output(Bool())
    val rx_data  = Output(UInt(8.W))
  })

  setInline(
    "SCUReadDPI.v",
    s"""
       |import "DPI-C" context function void scu_uart_rx_sample(
       |  input  int unsigned hart_id,
       |  input  int unsigned pop,
       |  output int unsigned valid,
       |  output int unsigned data
       |);
       |
       |module SCUReadDPI(
       |  input         clock,
       |  input         reset,
       |  input  [31:0] hart_id,
       |  input         enable,
       |  input         pop,
       |  output reg       rx_valid,
       |  output reg [7:0] rx_data
       |);
       |  integer valid;
       |  integer data;
       |  always @(posedge clock) begin
       |    if (reset) begin
       |      rx_valid <= 1'b0;
       |      rx_data <= 8'h0;
       |    end else if (enable) begin
       |      scu_uart_rx_sample(hart_id, {31'h0, pop}, valid, data);
       |      if (pop) begin
       |        rx_valid <= 1'b0;
       |      end else begin
       |        rx_valid <= (valid != 0);
       |        rx_data <= data[7:0];
       |      end
       |    end
       |  end
       |endmodule
    """.stripMargin
  )
}
