`timescale 1ns / 1ps
module tb_uart_loader;
  reg clk = 0;
  always #5 clk = ~clk;
  reg reset = 1, rx_valid = 0, rx_error = 0;
  reg [7:0] rx_data = 0;
  wire tx_valid;
  wire [7:0] tx_data;
  reg tx_ready = 1;
  integer tx_busy = 0, received = 0;
  reg [7:0] response[0:511];
  wire cpu_run, error_seen, load_en, load_valid;
  wire [11:0] load_addr;
  wire [127:0] load_data, load_result;
  wire [15:0] load_mask;
  uart_loader #(
      .RAM_BYTES (4096),
      .RX_TIMEOUT(64)
  ) dut (
      .*
  );
  axi_bram #(
      .WORDS(256)
  ) memory (
      .clk(clk),
      .reset(reset),
      .run(cpu_run),
      .arid(4'd0),
      .araddr(32'd0),
      .arlen(8'd0),
      .arsize(3'd4),
      .arburst(2'd1),
      .arvalid(1'b0),
      .arready(),
      .rid(),
      .rdata(),
      .rresp(),
      .rlast(),
      .rvalid(),
      .rready(1'b0),
      .awid(4'd0),
      .awaddr(32'd0),
      .awlen(8'd0),
      .awsize(3'd4),
      .awburst(2'd1),
      .awvalid(1'b0),
      .awready(),
      .wdata(128'd0),
      .wstrb(16'd0),
      .wlast(1'b0),
      .wvalid(1'b0),
      .wready(),
      .bid(),
      .bresp(),
      .bvalid(),
      .bready(1'b0),
      .load_en(load_en),
      .load_addr(load_addr),
      .load_data(load_data),
      .load_mask(load_mask),
      .load_valid(load_valid),
      .load_result(load_result)
  );
  always @(posedge clk) begin
    if (tx_valid && tx_ready) begin
      response[received] = tx_data;
      received = received + 1;
      tx_ready <= 0;
      tx_busy  <= 3;
    end else if (tx_busy != 0) tx_busy <= tx_busy - 1;
    else tx_ready <= 1;
  end
  function automatic [15:0] crc_step(input [15:0] c0, input [7:0] value);
    reg [15:0] c;
    begin
      c = c0;
      for (integer n = 7; n >= 0; n = n - 1) c = (c << 1) ^ ((c[15] ^ value[n]) ? 16'h1021 : 16'd0);
      crc_step = c;
    end
  endfunction
  task automatic byte_send(input [7:0] value);
    begin
      @(negedge clk);
      rx_data  = value;
      rx_valid = 1;
      @(negedge clk);
      rx_valid = 0;
    end
  endtask
  task automatic request(input [7:0] op, input [31:0] address, input [15:0] length,
                         input bit corrupt);
    reg [15:0] crc;
    reg [ 7:0] value;
    begin
      received = 0;
      crc = 16'hffff;
      byte_send(8'ha5);
      byte_send(8'h5a);
      byte_send(op);
      crc = crc_step(crc, op);
      byte_send(8'h37);
      crc = crc_step(crc, 8'h37);
      for (integer n = 0; n < 4; n = n + 1) begin
        value = address >> (n * 8);
        byte_send(value);
        crc = crc_step(crc, value);
      end
      byte_send(length[7:0]);
      crc = crc_step(crc, length[7:0]);
      byte_send(length[15:8]);
      crc = crc_step(crc, length[15:8]);
      if (op == 1)
        for (integer n = 0; n < length; n = n + 1) begin
          value = (n * 7 + 3) & 255;
          byte_send(value);
          crc = crc_step(crc, value);
        end
      byte_send(crc[7:0] ^ (corrupt ? 8'h01 : 8'h00));
      byte_send(crc[15:8]);
    end
  endtask
  task automatic reply(input [7:0] op, input [7:0] status, input integer length);
    reg [15:0] crc;
    begin
      while (received < length + 9) @(negedge clk);
      if(response[0]!==8'h5a || response[1]!==8'ha5 || response[2]!==(op|8'h80) ||
         response[3]!==8'h37 || response[4]!==status || {response[6],response[5]}!==length)
        $fatal(1, "Loader response header/status op=%h status=%h", op, response[4]);
      crc = 16'hffff;
      for (integer n = 2; n < length + 7; n = n + 1) crc = crc_step(crc, response[n]);
      if ({response[length+8], response[length+7]} !== crc) $fatal(1, "Loader response CRC");
      repeat (8) @(negedge clk);
    end
  endtask
  initial begin
    repeat (4) @(negedge clk);
    reset = 0;
    request(3, 32'h80000000, 0, 0);
    reply(3, 5, 0);
    if (cpu_run) $fatal(1, "RUN before loading accepted");
    request(4, 0, 0, 0);
    reply(4, 0, 12);
    if ({response[10], response[9], response[8], response[7]} !== 32'h01374150)
      $fatal(1, "INFO signature");
    request(1, 32'h80000003, 256, 0);
    reply(1, 0, 0);
    request(2, 32'h80000003, 256, 0);
    reply(2, 0, 256);
    for (integer n = 0; n < 256; n = n + 1)
    if (response[7+n] !== ((n * 7 + 3) & 255)) $fatal(1, "Loader roundtrip %d", n);
    // A corrupted frame targeting a shifted address must leave all data intact.
    request(1, 32'h80000004, 128, 1);
    reply(1, 1, 0);
    request(2, 32'h80000003, 256, 0);
    reply(2, 0, 256);
    for (integer n = 0; n < 256; n = n + 1)
    if (response[7+n] !== ((n * 7 + 3) & 255)) $fatal(1, "Bad CRC modified RAM");
    request(1, 32'h80000ff0, 32, 0);
    reply(1, 2, 0);
    request(8'h42, 0, 0, 0);
    reply(8'h42, 3, 0);
    byte_send(8'ha5);
    byte_send(8'h5a);
    byte_send(1);
    repeat (80) @(negedge clk);
    request(4, 0, 0, 0);
    reply(4, 0, 12);
    request(3, 32'h80000000, 0, 1);
    reply(3, 1, 0);
    if (cpu_run) $fatal(1, "Bad RUN CRC started CPU");
    request(3, 32'h80000000, 0, 0);
    reply(3, 0, 0);
    if (!cpu_run || !error_seen) $fatal(1, "RUN or error reporting");
    $display(
        "PASS: UART loader CRC, 256-byte unaligned RAM roundtrip, rejected writes/RUN, timeout and boot");
    $finish;
  end
  initial begin
    #1000000;
    $fatal(1, "Loader watchdog state=%d received=%d", dut.state, received);
  end
endmodule
