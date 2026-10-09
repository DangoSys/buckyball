// 8-N-1 receiver. Data is sampled in the middle of each bit after a verified
// start bit. A held-low/broken line reports one framing error, then waits high.
module uart_rx #(
    parameter integer DIVISOR = 434
) (
    input wire clk,
    reset,
    rx,
    output reg valid,
    output reg [7:0] data,
    output reg frame_error
);
  (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg rx_meta = 1, rx_sync = 1;
  always @(posedge clk) begin
    rx_meta <= rx;
    rx_sync <= rx_meta;
  end
  localparam integer WIDTH = $clog2(DIVISOR) < 1 ? 1 : $clog2(DIVISOR);
  localparam [2:0] IDLE = 0, START = 1, DATA = 2, STOP = 3, WAIT_HIGH = 4;
  reg [2:0] state, bit_index;
  reg [WIDTH-1:0] count;
  reg [7:0] shift;
  always @(posedge clk) begin
    valid <= 0;
    frame_error <= 0;
    if (reset) begin
      state <= IDLE;
      bit_index <= 0;
      count <= 0;
      shift <= 0;
      data <= 0;
    end else
      case (state)
        IDLE:
        if (!rx_sync) begin
          state <= START;
          count <= DIVISOR / 2 - 1;
        end
        START:
        if (count != 0) count <= count - 1'b1;
        else if (rx_sync) state <= IDLE;
        else begin
          state <= DATA;
          bit_index <= 0;
          count <= DIVISOR - 1;
        end
        DATA:
        if (count != 0) count <= count - 1'b1;
        else begin
          shift[bit_index] <= rx_sync;
          count <= DIVISOR - 1;
          if (bit_index == 7) state <= STOP;
          else bit_index <= bit_index + 1'b1;
        end
        STOP:
        if (count != 0) count <= count - 1'b1;
        else if (rx_sync) begin
          data  <= shift;
          valid <= 1;
          state <= IDLE;
        end else begin
          frame_error <= 1;
          state <= WAIT_HIGH;
        end
        WAIT_HIGH: if (rx_sync) state <= IDLE;
        default:   state <= IDLE;
      endcase
  end
endmodule
