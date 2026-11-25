// SubROB expansion port of GemminiBall: loop configuration and loop triggers.
interface gemmini_expansion_if (
    input logic clock,
    input logic reset
);
  logic                    command_ready;
  logic                    command_valid;
  logic [`BB_ROB_ID_W-1:0] command_bits_rob_id;
  logic [             6:0] command_bits_funct;
  logic [            63:0] command_bits_rs2;
  logic                    accepts;
  logic                    starts;
  logic                    complete_ready;
  logic                    complete_valid;
  logic [`BB_ROB_ID_W-1:0] complete_bits_rob_id;
  logic                    write_ready;
  logic                    write_valid;
  logic [             9:0] write_bits_addr;
  logic [             7:0] write_bits_data;
  // tb_top ties rows ready and read request ready to 1.
  logic                    rows_valid;
  logic                    rows_bits_last;
  logic                    read_req_valid;
  logic [            12:0] read_req_bits_addr;
  logic                    read_resp_ready;
  logic                    read_resp_valid;
  logic [             7:0] read_resp_bits_data;

  // The loop metadata bytes, as the MMIO bank would hold them: a request is
  // accepted at once and answered the next cycle, held until accepted.
  logic [             7:0] metadata             [136];
  always @(posedge clock) begin
    if (reset) begin
      read_resp_valid <= 1'b0;
    end else begin
      if (read_resp_valid && read_resp_ready) read_resp_valid <= 1'b0;
      if (read_req_valid) begin
        read_resp_valid     <= 1'b1;
        read_resp_bits_data <= metadata[read_req_bits_addr];
      end
    end
  end

  // The cycles of the last expansion command, for the perf tests: its
  // command handshake, and its completion handshake (a configuration) or the
  // handshake of its last row (a loop).
  longint unsigned perf_cycle;
  longint unsigned perf_accepted;
  longint unsigned perf_completed;
  always @(posedge clock) begin
    if (reset) begin
      perf_cycle <= 0;
    end else begin
      perf_cycle <= perf_cycle + 1;
      if (command_valid && command_ready) perf_accepted <= perf_cycle;
      if ((complete_valid && complete_ready) || (rows_valid && rows_bits_last))
        perf_completed <= perf_cycle;
    end
  end
endinterface
