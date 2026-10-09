`timescale 1ns / 1ps
module tb_uart_rx;
  reg clk = 0;
  always #5 clk = ~clk;
  reg reset = 1, rx = 1;
  wire valid, frame_error;
  wire [7:0] data;
  integer count = 0, errors = 0;
  reg [7:0] expected;
  uart_rx #(.DIVISOR(16)) dut (.*);
  always @(posedge clk) begin
    if (valid) begin
      if (data !== expected) $fatal(1, "UART expected %h got %h", expected, data);
      count = count + 1;
    end
    if (frame_error) errors = errors + 1;
  end
  task automatic bit_value(input bit value);
    begin
      rx = value;
      repeat (16) @(negedge clk);
    end
  endtask
  task automatic byte_value(input [7:0] value, input bit good_stop);
    begin
      bit_value(0);
      for (integer n = 0; n < 8; n = n + 1) bit_value(value[n]);
      bit_value(good_stop);
    end
  endtask
  initial begin
    repeat (4) @(negedge clk);
    reset = 0;
    repeat (4) @(negedge clk);
    // A short low glitch is not a valid start bit.
    rx = 0;
    repeat (3) @(negedge clk);
    rx = 1;
    repeat (32) @(negedge clk);
    if (count || errors) $fatal(1, "False-start rejection");
    for (integer n = 0; n < 256; n = n + 1) begin
      expected = n;
      byte_value(n, 1);
      repeat (3) @(negedge clk);
    end
    if (count != 256) $fatal(1, "Missing UART bytes %d", count);
    byte_value(8'h12, 0);
    repeat (160) @(negedge clk);
    if (count != 256 || errors != 1) $fatal(1, "Framing or break detection");
    bit_value(1);
    expected = 8'h5a;
    byte_value(expected, 1);
    repeat (8) @(negedge clk);
    if (count != 257) $fatal(1, "Recovery after framing error");
    $display("PASS: UART all byte values, false starts, framing errors and recovery");
    $finish;
  end
  initial begin
    #1000000;
    $fatal(1, "UART test watchdog");
  end
endmodule
