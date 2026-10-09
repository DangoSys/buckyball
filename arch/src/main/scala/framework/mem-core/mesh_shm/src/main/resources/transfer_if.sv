interface transfer_if (
    input logic clock
);
  logic reset;
  logic valid, ready, complete_valid, complete_ready, error;
  logic [7:0] source_core, target_core, tag, complete_tag;
  logic [9:0] source_bank, target_bank;
  logic [15:0] source_row, target_row;
  logic [16:0] rows;
  logic [4:0] request_valid, request_ready, response_valid, response_ready, write, response_error;
  logic [ 9:0] bank[5];
  logic [15:0] row [5];
  logic [127:0] data[5], response_data[5];
  logic [15:0] keep[5];
  logic [7:0] request_tag[5], response_tag[5];
  bit [127:0] memory[5][16][2048];
  bit initialized[5][16][2048];
  integer cycle = 0;
  for (genvar i = 0; i < 5; i++) begin
    assign request_ready[i] = !response_valid[i] && (cycle % 4 != 0);
    always @(posedge clock) begin
      if (reset) response_valid[i] <= 0;
      else if (request_valid[i] && request_ready[i]) begin
        response_valid[i] <= 1;
        response_tag[i]   <= request_tag[i];
        response_error[i] <= bank[i] >= 16 || row[i] >= 2048;
        response_data[i]  <= 0;
        if (bank[i] < 16 && row[i] < 2048) begin
          if (write[i]) begin
            for (int byte_index = 0; byte_index < 16; byte_index++)
            if (keep[i][byte_index])
              memory[i][bank[i]][row[i]][byte_index*8+:8] <= data[i][byte_index*8+:8];
            initialized[i][bank[i]][row[i]] <= 1;
          end else begin
            if (!initialized[i][bank[i]][row[i]]) $fatal(1, "uninitialized private-bank read");
            response_data[i] <= memory[i][bank[i]][row[i]];
          end
        end
      end else if (response_valid[i] && response_ready[i]) response_valid[i] <= 0;
    end
  end
  always @(posedge clock) cycle <= cycle + 1;
endinterface
