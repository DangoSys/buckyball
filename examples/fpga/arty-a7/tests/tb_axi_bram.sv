`timescale 1ns / 1ps
module tb_axi_bram;
  reg clk = 0;
  always #5 clk = ~clk;
  reg reset = 1, run = 0;
  reg [3:0] arid = 0, awid = 0;
  reg [31:0] araddr = 0, awaddr = 0;
  reg [7:0] arlen = 0, awlen = 0;
  reg [2:0] arsize = 4, awsize = 4;
  reg [1:0] arburst = 1, awburst = 1;
  reg arvalid = 0, awvalid = 0, wvalid = 0, wlast = 0, bready = 0, rready = 0;
  wire arready, awready, wready, rvalid, rlast, bvalid;
  wire [3:0] rid, bid;
  wire [1:0] rresp, bresp;
  wire [127:0] rdata;
  reg [127:0] wdata = 0;
  reg [15:0] wstrb = 0;
  reg load_en = 0;
  reg [11:0] load_addr = 0;
  reg [127:0] load_data = 0;
  reg [15:0] load_mask = 0;
  wire load_valid;
  wire [127:0] load_result;
  reg [127:0] reference[0:255];
  axi_bram #(.WORDS(256)) dut (.*);

  task automatic put(input integer address, input integer beats, input [2:0] size,
                     input [1:0] burst, input [15:0] mask, input [127:0] value,
                     input [1:0] response);
    begin
      @(negedge clk);
      awaddr = address;
      awlen = beats - 1;
      awsize = size;
      awburst = burst;
      awid = 9;
      awvalid = 1;
      @(posedge clk);
      while (!awready) @(posedge clk);
      @(negedge clk);
      awvalid = 0;
      for (integer n = 0; n < beats; n = n + 1) begin
        repeat (n % 3) @(negedge clk);
        wvalid = 1;
        wdata  = value + n;
        wstrb  = mask;
        wlast  = n == beats - 1;
        @(posedge clk);
        while (!wready) @(posedge clk);
        @(negedge clk);
        wvalid = 0;
      end
      while (!bvalid) @(negedge clk);
      repeat (3) begin
        if (bid !== 9 || bresp !== response || !bvalid) $fatal(1, "Write response/ID/backpressure");
        @(negedge clk);
      end
      bready = 1;
      @(negedge clk);
      bready = 0;
    end
  endtask

  task automatic get(input integer address, input integer beats, input [2:0] size,
                     input [1:0] burst, input [1:0] response);
    integer index;
    reg [127:0] snapshot;
    begin
      @(negedge clk);
      araddr = address;
      arlen = beats - 1;
      arsize = size;
      arburst = burst;
      arid = 6;
      arvalid = 1;
      @(posedge clk);
      while (!arready) @(posedge clk);
      @(negedge clk);
      arvalid = 0;
      for (integer n = 0; n < beats; n = n + 1) begin
        while (!rvalid) @(negedge clk);
        index = ((address - 32'h80000000) + (burst == 1 ? (n << size) : 0)) >> 4;
        if (rid !== 6 || rresp !== response || rlast !== (n == beats - 1))
          $fatal(1, "Read response/ID/last");
        if (rdata !== (response == 0 ? reference[index] : 128'd0))
          $fatal(
              1, "Read data at %h beat %0d got %h expected %h", address, n, rdata, reference[index]
          );
        snapshot = rdata;
        repeat ((n % 3) + 1) begin
          @(negedge clk);
          if (!rvalid || rdata !== snapshot) $fatal(1, "Read payload changed under backpressure");
        end
        rready = 1;
        @(negedge clk);
        rready = 0;
      end
    end
  endtask

  initial begin
    repeat (4) @(negedge clk);
    reset = 0;
    for (integer n = 0; n < 256; n = n + 1) begin
      reference[n] = {32'h12345678, 32'h89abcdef, 32'hfedcba98, 32'h76543210} + n;
      load_en = 1;
      load_addr = n * 16;
      load_mask = 16'hffff;
      load_data = reference[n];
      @(negedge clk);
    end
    load_en = 0;
    load_mask = 0;
    run = 1;
    get(32'h80000000, 16, 4, 1, 0);
    put(32'h80000040, 4, 4, 1, 16'hffff, 128'hdeadbeefcafebaab, 0);
    for (integer n = 0; n < 4; n = n + 1) reference[4+n] = 128'hdeadbeefcafebaab + n;
    get(32'h80000040, 4, 4, 1, 0);
    put(32'h80000054, 1, 2, 1, 16'h00f0, 128'h1122334455667788, 0);
    reference[5][63:32] = 32'h11223344;
    get(32'h80000050, 1, 4, 1, 0);
    get(32'h80000054, 1, 2, 1, 0);
    put(32'h80000060, 3, 4, 0, 16'hffff, 128'habc000, 0);
    reference[6] = 128'habc002;
    get(32'h80000060, 3, 4, 0, 0);
    // Rejected range/alignment/burst requests must never corrupt memory.
    put(32'h80000ff0, 2, 4, 1, 16'hffff, 128'hbad, 2);
    get(32'h80000ff0, 2, 4, 1, 2);
    get(32'h80001000, 1, 4, 1, 2);
    get(32'h80000003, 1, 2, 1, 2);
    get(32'h80000000, 4, 4, 2, 2);
    put(32'h80000004, 1, 2, 1, 16'hffff, 128'hbad, 2);
    get(32'h80000000, 1, 4, 1, 0);
    get(32'h80000ff0, 1, 4, 1, 0);
    @(negedge clk);
    reset = 1;
    repeat (3) @(negedge clk);
    reset = 0;
    get(32'h80000040, 4, 4, 1, 0);
    @(negedge clk);
    run = 0;
    load_en = 1;
    load_addr = 16'h50;
    load_mask = 0;
    repeat (2) @(negedge clk);
    if (!load_valid || load_result !== reference[5]) $fatal(1, "Loader readback");
    if (arready || awready || wready) $fatal(1, "CPU accesses permitted while halted");
    $display("PASS: AXI BRAM bursts, strobes, IDs, stalls, errors, reset and loader readback");
    $finish;
  end
  initial begin
    #200000;
    $fatal(1, "AXI test watchdog");
  end
endmodule
