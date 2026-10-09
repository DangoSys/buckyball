// Arty main memory: 128-bit true dual-port BRAM, one read and one write burst.
// Loader owns port A only while the complete SoC is held in reset.
module axi_bram #(
    parameter integer ADDR_WIDTH = 32,
    parameter integer ID_WIDTH = 4,
    parameter integer WORDS = 8192,
    parameter logic [ADDR_WIDTH-1:0] BASE = 32'h80000000,
    parameter INIT_FILE = ""
) (
    input wire clk,
    reset,
    run,
    input wire [ID_WIDTH-1:0] arid,
    input wire [ADDR_WIDTH-1:0] araddr,
    input wire [7:0] arlen,
    input wire [2:0] arsize,
    input wire [1:0] arburst,
    input wire arvalid,
    output wire arready,
    output reg [ID_WIDTH-1:0] rid,
    output reg [127:0] rdata,
    output reg [1:0] rresp,
    output reg rlast,
    rvalid,
    input wire rready,
    input wire [ID_WIDTH-1:0] awid,
    input wire [ADDR_WIDTH-1:0] awaddr,
    input wire [7:0] awlen,
    input wire [2:0] awsize,
    input wire [1:0] awburst,
    input wire awvalid,
    output wire awready,
    input wire [127:0] wdata,
    input wire [15:0] wstrb,
    input wire wlast,
    wvalid,
    output wire wready,
    output reg [ID_WIDTH-1:0] bid,
    output reg [1:0] bresp,
    output reg bvalid,
    input wire bready,
    input wire load_en,
    input wire [$clog2(WORDS)+3:0] load_addr,
    input wire [127:0] load_data,
    input wire [15:0] load_mask,
    output reg load_valid,
    output wire [127:0] load_result
);
  localparam integer INDEX_WIDTH = $clog2(WORDS);
  (* ram_style = "block" *) reg [127:0] ram[0:WORDS-1];
  initial if (INIT_FILE != "") $readmemh(INIT_FILE, ram);
  reg [127:0] memory_read;
  assign load_result = memory_read;
  reg read_active, read_pending, read_error, write_active, write_error;
  reg [ADDR_WIDTH-1:0] read_address, write_address;
  reg [7:0] read_left, write_left;
  reg [2:0] read_size, write_size;
  reg [1:0] read_burst, write_burst;
  wire request_read = run && read_active && !rvalid && !read_pending;
  wire write_fire = wvalid && wready;
  wire [INDEX_WIDTH-1:0] read_index = (read_address - BASE) >> 4;
  wire [INDEX_WIDTH-1:0] write_index = (write_address - BASE) >> 4;
  wire port_a_enable = (!run && load_en) || (request_read && !read_error);
  wire [INDEX_WIDTH-1:0] port_a_address = run ? read_index : load_addr[INDEX_WIDTH+3:4];
  wire [15:0] port_a_write = !run ? load_mask : 16'd0;
  wire [31:0] permitted_mask = ((32'd1 << (1 << write_size)) - 1) << write_address[3:0];
  wire bad_strobe = |(wstrb & ~permitted_mask[15:0]);

  function automatic invalid_burst(input [ADDR_WIDTH-1:0] address, input [7:0] length,
                                   input [2:0] size, input [1:0] burst);
    reg [ADDR_WIDTH:0] last_byte;
    begin
      last_byte = {1'b0, address} + (((burst == 1 ? {1'b0, length} + 1 : 1) << size) - 1);
      invalid_burst = size > 4 || burst > 1 ||
        (address & ((1 << size)-1)) != 0 ||
        address < BASE || last_byte >= ({1'b0,BASE} + WORDS*16) ||
        (address >> 12) != (last_byte >> 12);
    end
  endfunction

  // No reset on RAM storage or its data register: preserves BRAM inference.
  always @(posedge clk) begin
    if (port_a_enable) begin
      memory_read <= ram[port_a_address];
      for (integer lane = 0; lane < 16; lane = lane + 1)
      if (port_a_write[lane]) ram[port_a_address][lane*8+:8] <= load_data[lane*8+:8];
    end
  end
  always @(posedge clk) begin
    if (write_fire && !write_error && !bad_strobe)
      for (integer lane = 0; lane < 16; lane = lane + 1)
      if (wstrb[lane]) ram[write_index][lane*8+:8] <= wdata[lane*8+:8];
  end

  assign arready = run && !reset && !read_active;
  assign awready = run && !reset && !write_active && !bvalid;
  assign wready  = run && !reset && write_active;

  always @(posedge clk) begin
    load_valid <= !reset && !run && load_en;
    if (reset || !run) begin
      read_active <= 0;
      read_pending <= 0;
      rvalid <= 0;
      rlast <= 0;
      rid <= 0;
      rdata <= 0;
      rresp <= 0;
      read_error <= 0;
      read_address <= 0;
      read_left <= 0;
      read_size <= 0;
      read_burst <= 0;
      write_active <= 0;
      write_error <= 0;
      write_address <= 0;
      write_left <= 0;
      write_size <= 0;
      write_burst <= 0;
      bid <= 0;
      bresp <= 0;
      bvalid <= 0;
    end else begin
      if (arvalid && arready) begin
        read_active <= 1;
        rid <= arid;
        read_address <= araddr;
        read_left <= arlen;
        read_size <= arsize;
        read_burst <= arburst;
        read_error <= invalid_burst(araddr, arlen, arsize, arburst);
      end
      read_pending <= request_read;
      if (read_pending) begin
        rvalid <= 1;
        rdata  <= read_error ? 128'd0 : memory_read;
        rresp  <= read_error ? 2'b10 : 2'b00;
        rlast  <= read_left == 0;
      end
      if (rvalid && rready) begin
        rvalid <= 0;
        if (read_left == 0) read_active <= 0;
        else begin
          read_left <= read_left - 1'b1;
          if (read_burst == 1) read_address <= read_address + (1 << read_size);
        end
      end
      if (bvalid && bready) bvalid <= 0;
      if (awvalid && awready) begin
        write_active <= 1;
        bid <= awid;
        write_address <= awaddr;
        write_left <= awlen;
        write_size <= awsize;
        write_burst <= awburst;
        write_error <= invalid_burst(awaddr, awlen, awsize, awburst);
      end
      if (write_fire) begin
        write_error <= write_error || bad_strobe || (wlast != (write_left == 0));
        if (write_left == 0 || wlast) begin
          write_active <= 0;
          bvalid <= 1;
          bresp <= (write_error || bad_strobe || (wlast != (write_left == 0))) ? 2'b10 : 2'b00;
        end else begin
          write_left <= write_left - 1'b1;
          if (write_burst == 1) write_address <= write_address + (1 << write_size);
        end
      end
    end
  end
endmodule
