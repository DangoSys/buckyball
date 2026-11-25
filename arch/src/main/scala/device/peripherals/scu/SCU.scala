package framework.device.peripherals.scu

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util.HasBlackBoxInline

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

  @public
  val io = IO(new Bundle {
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

  @public
  val io = IO(new Bundle {
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
