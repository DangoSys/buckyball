// Digilent Arty A7-100T. SW0 selects the resident demo at reset.
module pebble_a7_top (
    input wire CLK100MHZ,
    btn0,
    sw0,
    uart_txd_in,
    output wire uart_rxd_out,
    output wire [3:0] led
);
  wire clk_in, feedback, feedback_buf, clk_out, clk, locked;
  IBUF input_clock (
      .I(CLK100MHZ),
      .O(clk_in)
  );
  MMCME2_BASE #(
      .CLKIN1_PERIOD(10.0),
      .DIVCLK_DIVIDE(1),
      .CLKFBOUT_MULT_F(10.0),
      .CLKOUT0_DIVIDE_F(20.0),
      .STARTUP_WAIT("FALSE")
  ) clock_mmcm (
      .CLKIN1(clk_in),
      .CLKFBIN(feedback_buf),
      .CLKFBOUT(feedback),
      .CLKOUT0(clk_out),
      .LOCKED(locked),
      .RST(1'b0),
      .PWRDWN(1'b0),
      .CLKFBOUTB(),
      .CLKOUT0B(),
      .CLKOUT1(),
      .CLKOUT1B(),
      .CLKOUT2(),
      .CLKOUT2B(),
      .CLKOUT3(),
      .CLKOUT3B(),
      .CLKOUT4(),
      .CLKOUT5(),
      .CLKOUT6()
  );
  BUFG feedback_buffer (
      .I(feedback),
      .O(feedback_buf)
  );
  BUFG clock_buffer (
      .I(clk_out),
      .O(clk)
  );
  (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] lock_sync = 3'b111;
  always @(posedge clk or negedge locked)
    if (!locked) lock_sync <= 3'b111;
    else lock_sync <= {lock_sync[1:0], 1'b0};
  (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg btn_meta = 0, btn_sync = 0;
  (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg sw_meta = 0, sw_sync = 0;
  always @(posedge clk) begin
    btn_meta <= btn0;
    btn_sync <= btn_meta;
    sw_meta  <= sw0;
    sw_sync  <= sw_meta;
  end
  reg [17:0] reset_count = 0;
  reg reset = 1;
  always @(posedge clk) begin
    reset <= lock_sync[2] || btn_sync || !(&reset_count);
    if (lock_sync[2] || btn_sync) reset_count <= 0;
    else if (!(&reset_count)) reset_count <= reset_count + 1'b1;
  end
  wire cpu_run, exited, loader_error;
  wire [31:0] exit_code;
  pebble_a7_platform platform (
      .clk(clk),
      .reset(reset),
      .boot_demo(sw_sync),
      .uart_rx_in(uart_txd_in),
      .uart_tx_out(uart_rxd_out),
      .cpu_run(cpu_run),
      .exited(exited),
      .loader_error(loader_error),
      .exit_code(exit_code)
  );
  reg [25:0] heartbeat = 0;
  always @(posedge clk)
    if (reset) heartbeat <= 0;
    else heartbeat <= heartbeat + 1'b1;
  assign led = {
    heartbeat[25],
    cpu_run && !reset,
    loader_error || (exited && exit_code != 0),
    exited && exit_code == 0
  };
endmodule
