// Real AXI MMIO UART: TX 0x00, STATUS 0x04, RX 0x08, EXIT 0x10,
// byte-writeable scratch 0x18, cycle counter 0x20. 16-byte receive FIFO.
module axi_uart #(
    parameter integer ADDR_WIDTH = 32,
    ID_WIDTH = 2,
    parameter logic [ADDR_WIDTH-1:0] BASE = 32'h60000000
) (
    input wire clk,
    reset,
    input wire [ID_WIDTH-1:0] arid,
    awid,
    input wire [ADDR_WIDTH-1:0] araddr,
    awaddr,
    input wire [7:0] arlen,
    awlen,
    input wire [2:0] arsize,
    awsize,
    input wire [1:0] arburst,
    awburst,
    input wire arvalid,
    awvalid,
    wvalid,
    wlast,
    rready,
    bready,
    output wire arready,
    awready,
    wready,
    input wire [63:0] wdata,
    input wire [7:0] wstrb,
    output reg [ID_WIDTH-1:0] rid,
    bid,
    output reg [63:0] rdata,
    output reg [1:0] rresp,
    bresp,
    output reg rvalid,
    rlast,
    bvalid,
    input wire rx_valid,
    input wire [7:0] rx_data,
    input wire rx_error,
    output wire tx_valid,
    output wire [7:0] tx_data,
    input wire tx_ready,
    output reg exited,
    output reg [31:0] exit_code
);
  reg [7:0] fifo[0:15];
  reg [3:0] read_pointer, write_pointer;
  reg [4:0] count;
  reg overflow, framing_error, read_pop, write_active, write_error;
  reg [7:0] read_left, write_left;
  reg [11:0] write_offset;
  reg [63:0] scratch, cycles;
  wire pop = rvalid && rready && read_pop;
  wire push = rx_valid && (count < 16 || pop);
  wire needs_tx = !write_error && write_offset == 0 && wstrb[0];
  wire write_fire = wvalid && wready;
  wire read_in_range = araddr >= BASE && araddr < BASE + 4096;
  wire write_in_range = awaddr >= BASE && awaddr < BASE + 4096;
  assign arready  = !reset && !rvalid;
  assign awready  = !reset && !write_active && !bvalid;
  assign wready   = !reset && write_active && (!needs_tx || tx_ready);
  assign tx_valid = !reset && write_active && wvalid && needs_tx;
  assign tx_data  = wdata[7:0];

  always @(posedge clk) begin
    if (reset) begin
      read_pointer <= 0;
      write_pointer <= 0;
      count <= 0;
      overflow <= 0;
      framing_error <= 0;
      read_pop <= 0;
      write_active <= 0;
      write_error <= 0;
      read_left <= 0;
      write_left <= 0;
      write_offset <= 0;
      scratch <= 0;
      cycles <= 0;
      exited <= 0;
      exit_code <= 0;
      rid <= 0;
      bid <= 0;
      rdata <= 0;
      rresp <= 0;
      bresp <= 0;
      rvalid <= 0;
      rlast <= 0;
      bvalid <= 0;
    end else begin
      cycles <= cycles + 1'b1;
      if (rvalid && rready) begin
        read_pop <= 0;
        if (read_left == 0) rvalid <= 0;
        else begin
          read_left <= read_left - 1'b1;
          rlast <= read_left == 1;
        end
      end
      if (arvalid && arready) begin
        rvalid <= 1;
        rid <= arid;
        rresp <= 0;
        rlast <= arlen == 0;
        read_left <= arlen;
        read_pop <= 0;
        rdata <= 0;
        if(!read_in_range || arlen!=0 || arsize>3 || arburst>1 || (araddr & ((1<<arsize)-1))!=0)
          rresp <= 2;
        else
          case (araddr[11:0])
            0: rdata <= 0;
            4: rdata <= {28'd0, framing_error, overflow, (count != 0), tx_ready, 32'd0};
            8: begin
              rdata <= count != 0 ? {32'd0, 1'b1, 23'd0, fifo[read_pointer]} : 64'd0;
              read_pop <= count != 0;
            end
            16: rdata <= {31'd0, exited, exit_code};
            24: rdata <= scratch;
            32: rdata <= cycles;
            default: rresp <= 2;
          endcase
      end
      if (bvalid && bready) bvalid <= 0;
      if (awvalid && awready) begin
        write_active <= 1;
        bid <= awid;
        write_left <= awlen;
        write_offset <= awaddr[11:0];
        write_error<=!write_in_range || awlen!=0 || awsize>3 || awburst>1 ||
          (awaddr & ((1<<awsize)-1))!=0 ||
          !(awaddr[11:0]==0 || awaddr[11:0]==4 || awaddr[11:0]==16 || awaddr[11:3]==3);
      end
      if (write_fire) begin
        if (!write_error)
          case (write_offset)
            4:
            if (wstrb[4]) begin
              if (wdata[34]) overflow <= 0;
              if (wdata[35]) framing_error <= 0;
            end
            16: begin
              for (integer n = 0; n < 4; n = n + 1)
              if (wstrb[n]) exit_code[n*8+:8] <= wdata[n*8+:8];
              if (|wstrb[3:0]) exited <= 1;
            end
            default:
            if (write_offset[11:3] == 3)
              for (integer n = 0; n < 8; n = n + 1) if (wstrb[n]) scratch[n*8+:8] <= wdata[n*8+:8];
          endcase
        if (write_left == 0 || wlast) begin
          write_active <= 0;
          bvalid <= 1;
          bresp <= write_error || (wlast != (write_left == 0)) ? 2'b10 : 2'b00;
        end else write_left <= write_left - 1'b1;
      end
      if (push) begin
        fifo[write_pointer] <= rx_data;
        write_pointer <= write_pointer + 1'b1;
      end
      if (pop) read_pointer <= read_pointer + 1'b1;
      case ({
        push, pop
      })
        2'b10:   count <= count + 1'b1;
        2'b01:   count <= count - 1'b1;
        default: count <= count;
      endcase
      if (rx_valid && !push) overflow <= 1;
      if (rx_error) framing_error <= 1;
    end
  end
endmodule
