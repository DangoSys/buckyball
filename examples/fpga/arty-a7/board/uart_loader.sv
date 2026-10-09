// Stop-and-wait serial boot protocol. Requests: A5 5A op seq addrLE32 lenLE16
// [WRITE payload] crcLE16. CRC-16/CCITT-FALSE covers op through payload.
// Responses: 5A A5 (op|80) seq status lenLE16 [data] crcLE16.
// CRC is checked before any RAM write; CPU starts after the RUN reply drains.
module uart_loader #(
    parameter integer RAM_BYTES  = 131072,
    parameter integer CLOCK_HZ   = 50000000,
    parameter integer RX_TIMEOUT = 5000000
) (
    input wire clk,
    reset,
    input wire rx_valid,
    input wire [7:0] rx_data,
    input wire rx_error,
    output wire tx_valid,
    output reg [7:0] tx_data,
    input wire tx_ready,
    output reg cpu_run,
    output reg error_seen,
    output wire load_en,
    output wire [$clog2(RAM_BYTES)-1:0] load_addr,
    output reg [127:0] load_data,
    output reg [15:0] load_mask,
    input wire load_valid,
    input wire [127:0] load_result
);
  localparam [3:0] SYNC0=0,SYNC1=1,HEADER=2,PAYLOAD=3,CRC0=4,CRC1=5,
    EXECUTE=6,WRITE=7,READ_REQ=8,READ_WAIT=9,REPLY=10,DRAIN=11;
  reg [3:0] state;
  reg [7:0] operation, sequence_id, status;
  reg [31:0] address;
  reg [15:0] length, reply_length, crc, received_crc;
  reg [8:0] cursor, reply_index;
  reg [3:0] header_index;
  reg [7:0] payload[0:255];
  reg pending_run, image_loaded;
  reg [$clog2(RX_TIMEOUT+1)-1:0] timeout_count;
  wire receiving = state==SYNC1 || state==HEADER || state==PAYLOAD || state==CRC0 || state==CRC1;
  wire [32:0] end_address = {1'b0, address} + length;
  wire valid_range = length!=0 && length<=256 && address>=32'h80000000 &&
    end_address<=33'h080000000+RAM_BYTES;
  wire [31:0] byte_address = address + cursor;
  assign load_addr = byte_address[$clog2(RAM_BYTES)-1:0];
  assign load_en   = !cpu_run && (state == WRITE || state == READ_REQ);
  always @* begin
    load_data = 0;
    load_mask = 0;
    if (state == WRITE) begin
      load_data[byte_address[3:0]*8+:8] = payload[cursor[7:0]];
      load_mask = 16'b1 << byte_address[3:0];
    end
  end
  function automatic [15:0] crc_next(input [15:0] old_crc, input [7:0] value);
    reg [15:0] c;
    begin
      c = old_crc ^ {value, 8'd0};
      for (integer n = 0; n < 8; n = n + 1) c = c[15] ? (c << 1) ^ 16'h1021 : c << 1;
      crc_next = c;
    end
  endfunction
  assign tx_valid = state == REPLY;
  always @* begin
    case (reply_index)
      0: tx_data = 8'h5a;
      1: tx_data = 8'ha5;
      2: tx_data = operation | 8'h80;
      3: tx_data = sequence_id;
      4: tx_data = status;
      5: tx_data = reply_length[7:0];
      6: tx_data = reply_length[15:8];
      default: begin
        if (reply_index < 7 + reply_length) tx_data = payload[reply_index-7];
        else if (reply_index == 7 + reply_length) tx_data = crc[7:0];
        else tx_data = crc[15:8];
      end
    endcase
  end
  always @(posedge clk) begin
    if (reset) begin
      state <= SYNC0;
      cpu_run <= 0;
      error_seen <= 0;
      image_loaded <= 0;
      pending_run <= 0;
      operation <= 0;
      sequence_id <= 0;
      status <= 0;
      address <= 0;
      length <= 0;
      reply_length <= 0;
      crc <= 16'hffff;
      received_crc <= 0;
      cursor <= 0;
      reply_index <= 0;
      header_index <= 0;
      timeout_count <= 0;
    end else if (!cpu_run) begin
      if (!receiving || rx_valid) timeout_count <= 0;
      else timeout_count <= timeout_count + 1'b1;
      if ((receiving && timeout_count == RX_TIMEOUT - 1) || (receiving && rx_error)) begin
        state <= SYNC0;
        error_seen <= 1;
        timeout_count <= 0;
      end else
        case (state)
          SYNC0: if (rx_valid && rx_data == 8'ha5) state <= SYNC1;
          SYNC1:
          if (rx_valid) begin
            if (rx_data == 8'h5a) begin
              state <= HEADER;
              header_index <= 0;
              crc <= 16'hffff;
              length <= 0;
              address <= 0;
              pending_run <= 0;
              reply_index <= 0;
              reply_length <= 0;
              status <= 0;
            end else if (rx_data != 8'ha5) state <= SYNC0;
          end
          HEADER:
          if (rx_valid) begin
            crc <= crc_next(crc, rx_data);
            case (header_index)
              0: operation <= rx_data;
              1: sequence_id <= rx_data;
              2: address[7:0] <= rx_data;
              3: address[15:8] <= rx_data;
              4: address[23:16] <= rx_data;
              5: address[31:24] <= rx_data;
              6: length[7:0] <= rx_data;
              7: begin
                length[15:8] <= rx_data;
                cursor <= 0;
                if ({rx_data, length[7:0]} > 256) begin
                  state <= SYNC0;
                  error_seen <= 1;
                end else if (operation == 1 && {rx_data, length[7:0]} != 0) state <= PAYLOAD;
                else state <= CRC0;
              end
            endcase
            header_index <= header_index + 1'b1;
          end
          PAYLOAD:
          if (rx_valid) begin
            payload[cursor[7:0]] <= rx_data;
            crc <= crc_next(crc, rx_data);
            cursor <= cursor + 1'b1;
            if (cursor + 1 == length) state <= CRC0;
          end
          CRC0:
          if (rx_valid) begin
            received_crc[7:0] <= rx_data;
            state <= CRC1;
          end
          CRC1:
          if (rx_valid) begin
            cursor <= 0;
            if ({rx_data, received_crc[7:0]} != crc) begin
              status <= 1;
              error_seen <= 1;
              crc <= 16'hffff;
              state <= REPLY;
            end else state <= EXECUTE;
          end
          EXECUTE: begin
            crc <= 16'hffff;
            if (operation == 1 || operation == 2) begin
              if (!valid_range) begin
                status <= 2;
                error_seen <= 1;
                state <= REPLY;
              end else if (operation == 1) state <= WRITE;
              else begin
                reply_length <= length;
                state <= READ_REQ;
              end
            end else if (operation == 3 && length == 0 && address == 32'h80000000) begin
              if (!image_loaded) begin
                status <= 5;
                error_seen <= 1;
              end else pending_run <= 1;
              state <= REPLY;
            end else if (operation == 4 && length == 0 && address == 0) begin
              payload[0] <= 8'h50;
              payload[1] <= 8'h41;
              payload[2] <= 8'h37;
              payload[3] <= 1;
              for (integer n = 0; n < 4; n = n + 1) begin
                payload[4+n] <= (RAM_BYTES >> (n * 8)) & 255;
                payload[8+n] <= (CLOCK_HZ >> (n * 8)) & 255;
              end
              reply_length <= 12;
              state <= REPLY;
            end else begin
              status <= 3;
              error_seen <= 1;
              state <= REPLY;
            end
          end
          WRITE: begin
            if (cursor + 1 == length) begin
              image_loaded <= 1;
              state <= REPLY;
            end else cursor <= cursor + 1'b1;
          end
          READ_REQ: state <= READ_WAIT;
          READ_WAIT:
          if (load_valid) begin
            payload[cursor[7:0]] <= load_result[byte_address[3:0]*8+:8];
            if (cursor + 1 == length) state <= REPLY;
            else begin
              cursor <= cursor + 1'b1;
              state  <= READ_REQ;
            end
          end
          REPLY:
          if (tx_ready) begin
            if (reply_index >= 2 && reply_index < 7 + reply_length) crc <= crc_next(crc, tx_data);
            if (reply_index == 8 + reply_length) state <= DRAIN;
            else reply_index <= reply_index + 1'b1;
          end
          DRAIN:
          if (tx_ready) begin
            if (pending_run) cpu_run <= 1;
            state <= SYNC0;
          end
          default: state <= SYNC0;
        endcase
    end
  end
endmodule
