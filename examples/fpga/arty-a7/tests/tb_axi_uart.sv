`timescale 1ns / 1ps
module tb_axi_uart;
  reg clk = 0;
  always #5 clk = ~clk;
  reg reset = 1;
  reg [1:0] arid = 2, awid = 3;
  reg [31:0] araddr = 0, awaddr = 0;
  reg [7:0] arlen = 0, awlen = 0;
  reg [2:0] arsize = 2, awsize = 2;
  reg [1:0] arburst = 1, awburst = 1;
  reg arvalid = 0, awvalid = 0, wvalid = 0, wlast = 1, rready = 0, bready = 0;
  wire arready, awready, wready;
  reg [63:0] wdata = 0;
  reg [ 7:0] wstrb = 0;
  wire [1:0] rid, bid, rresp, bresp;
  wire [63:0] rdata;
  wire rvalid, rlast, bvalid;
  reg rx_valid = 0, rx_error = 0, tx_ready = 1;
  reg [7:0] rx_data = 0;
  wire tx_valid;
  wire [7:0] tx_data;
  wire exited;
  wire [31:0] exit_code;
  integer transmitted = 0;
  axi_uart dut (.*);
  always @(posedge clk)
    if (tx_valid && tx_ready) begin
      if (tx_data !== 8'h51) $fatal(1, "TX data");
      transmitted = transmitted + 1;
    end
  task automatic put(input [31:0] addr, input [2:0] size, input [7:0] mask, input [63:0] value,
                     input [1:0] response);
    @(negedge clk);
    awaddr  = addr;
    awsize  = size;
    awvalid = 1;
    @(posedge clk);
    while (!awready) @(posedge clk);
    @(negedge clk);
    awvalid = 0;
    wdata   = value;
    wstrb   = mask;
    wvalid  = 1;
    @(posedge clk);
    while (!wready) @(posedge clk);
    @(negedge clk);
    wvalid = 0;
    while (!bvalid) @(negedge clk);
    repeat (4) begin
      if (bresp !== response || bid !== 3 || !bvalid) $fatal(1, "B response/backpressure");
      @(negedge clk);
    end
    bready = 1;
    @(negedge clk);
    bready = 0;
  endtask
  task automatic get(input [31:0] addr, input [2:0] size, input [63:0] value, input [1:0] response);
    @(negedge clk);
    araddr  = addr;
    arsize  = size;
    arvalid = 1;
    @(posedge clk);
    while (!arready) @(posedge clk);
    @(negedge clk);
    arvalid = 0;
    while (!rvalid) @(negedge clk);
    repeat (4) begin
      if (rresp !== response || rid !== 2 || !rlast || !rvalid || rdata !== value)
        $fatal(1, "R response at %h got %h expected %h resp %d", addr, rdata, value, rresp);
      @(negedge clk);
    end
    rready = 1;
    @(negedge clk);
    rready = 0;
  endtask
  task automatic receive_byte(input [7:0] value);
    @(negedge clk);
    rx_data  = value;
    rx_valid = 1;
    @(negedge clk);
    rx_valid = 0;
  endtask
  initial begin
    repeat (4) @(negedge clk);
    reset = 0;
    get('h60000004, 2, 64'h0000000100000000, 0);
    put('h60000018, 3, 'hff, 64'h123456789abcdef0, 0);
    put('h6000001b, 0, 'h08, 64'h0000000055000000, 0);
    get('h60000018, 3, 64'h1234567855bcdef0, 0);
    fork
      begin
        tx_ready = 0;
        put('h60000000, 0, 1, 'h51, 0);
      end
      begin
        repeat (12) @(negedge clk);
        if (transmitted != 0) $fatal(1, "TX ignored backpressure");
        tx_ready = 1;
      end
    join
    if (transmitted != 1) $fatal(1, "TX count");
    for (integer n = 0; n < 17; n = n + 1) receive_byte(n + 32);
    rx_error = 1;
    @(negedge clk);
    rx_error = 0;
    get('h60000004, 2, 64'h0000000f00000000, 0);
    for (integer n = 0; n < 16; n = n + 1) get('h60000008, 2, 64'h80000020 + n, 0);
    get('h60000008, 2, 0, 0);
    put('h60000004, 2, 'hf0, 64'h0000000c00000000, 0);
    get('h60000004, 2, 64'h0000000100000000, 0);
    put('h60000010, 2, 'h0f, 7, 0);
    if (!exited || exit_code != 7) $fatal(1, "Exit register");
    put('h60000010, 2, 'h0f, 0, 0);
    get('h60000010, 3, 64'h100000000, 0);
    get('h50000000, 2, 0, 2);
    get('h60000005, 2, 0, 2);
    put('h60000030, 2, 'h0f, 123, 2);
    get('h60000018, 3, 64'h1234567855bcdef0, 0);
    reset = 1;
    @(negedge clk);
    reset = 0;
    get('h60000018, 3, 0, 0);
    $display("PASS: AXI UART byte lanes, TX stalls, RX FIFO/overflow, errors, exit and reset");
    $finish;
  end
  initial begin
    #100000;
    $fatal(1, "AXI UART timeout");
  end
endmodule
