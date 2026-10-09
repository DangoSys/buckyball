module uart_tx #(
    parameter integer DIVISOR = 434  // 50 MHz / 115200, error < 0.01%.
) (
    input wire clk,
    input wire reset,
    input wire valid,
    input wire [7:0] data,
    output wire ready,
    output wire tx
);
  localparam integer COUNT_BITS = $clog2(DIVISOR) < 1 ? 1 : $clog2(DIVISOR);
  reg [COUNT_BITS-1:0] count;
  reg [9:0] shift;
  reg [3:0] bits_left;
  assign ready = bits_left == 0;
  assign tx = shift[0];

  always @(posedge clk) begin
    if (reset) begin
      count <= 0;
      shift <= 10'h3ff;
      bits_left <= 0;
    end else if (ready) begin
      if (valid) begin
        shift <= {1'b1, data, 1'b0};
        bits_left <= 10;
        count <= DIVISOR - 1;
      end
    end else if (count == 0) begin
      shift <= {1'b1, shift[9:1]};
      bits_left <= bits_left - 1'b1;
      count <= DIVISOR - 1;
    end else begin
      count <= count - 1'b1;
    end
  end
endmodule
