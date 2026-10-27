module redmule_reference (
    input logic clock,
    reset
);
  logic control_req = 0, control_wen;
  logic [31:0] control_addr, control_data, control_result;
  logic control_grant, control_valid, busy, event_done;
  logic request, load, ready;
  logic [31:0] address, enables;
  logic [255:0] write_data, response_data;
  logic [7:0] request_id, response_id;
  logic response_valid = 0;
  bit [127:0] memory[3][1 << `BB_BANK_ADDR_W];
  int rows, reduction, row_base;
  longint ticks = 0;

  RedmuleNative original (
      .clk_i(clock),
      .rst_ni(!reset),
      .busy_o(busy),
      .evt_o(event_done),
      .ctrl_req_i(control_req),
      .ctrl_wen_i(control_wen),
      .ctrl_addr_i(control_addr),
      .ctrl_data_i(control_data),
      .ctrl_gnt_o(control_grant),
      .ctrl_valid_o(control_valid),
      .ctrl_data_o(control_result),
      .mem_req_o(request),
      .mem_wen_o(load),
      .mem_addr_o(address),
      .mem_be_o(enables),
      .mem_data_o(write_data),
      .mem_id_o(request_id),
      .mem_gnt_i(1'b1),
      .mem_valid_i(response_valid),
      .mem_data_i(response_data),
      .mem_id_i(response_id),
      .mem_ready_o(ready)
  );

  function void load_word(int bank, int row, bit [127:0] data);
    memory[bank][row] = data;
  endfunction

  always @(posedge clock) begin
    int bank, word_index, limit;
    ticks <= ticks + 1;
    response_valid <= 0;
    if (!reset && request) begin
      bank = int'(address[25:24]) - 1;
      word_index = address[23:4];
      case (bank)
        0: limit = row_base + rows * reduction / 8;
        1: limit = row_base + reduction;
        2: limit = row_base + rows;
        default: $fatal(1, "Native reference address has no bank region");
      endcase
      if (load) begin
        response_data <= {
          word_index + 1 < limit ? memory[bank][word_index+1] : 128'b0,
          word_index < limit ? memory[bank][word_index] : 128'b0
        };
        response_id <= request_id;
        response_valid <= 1;
      end else begin
        if (bank != 2) $fatal(1, "Native reference store is outside C region");
        for (int byte_index = 0; byte_index < 32; byte_index++)
        if (enables[byte_index] && word_index + byte_index / 16 < limit)
          memory[2][word_index+byte_index/16][(byte_index%16)*8+:8] <= write_data[byte_index*8+:8];
      end
    end
  end

  task execute(int m, int k, int base, bit first, output longint elapsed);
    longint started;
    int register_addr[9] = '{4, 44, 48, 52, 32, 36, 40, 60, 0};
    int register_data[9];
    rows = m;
    reduction = k;
    row_base = base;
    register_data = '{
        0,
        32'h01000000 + (base << 4),
        32'h02000000 + (base << 4),
        32'h03000000 + (base << 4),
        (8 << 16) | m,
        32'h02800000 | (!first << 20) | k,
        0,
        0,
        0
    };
    started = ticks;
    for (int step = 0; step < 9; step++) begin
      @(negedge clock);
      control_addr = register_addr[step];
      control_data = register_data[step];
      control_wen  = (step == 0);
      control_req  = 1;
      do @(posedge clock); while (!control_grant);
      @(negedge clock);
      control_req = 0;
      wait (control_valid);
      if (step == 0 && control_result == 32'hffffffff)
        $fatal(1, "Native reference has no job context");
    end
    wait (event_done);
    wait (!busy);
    elapsed = ticks - started;
    @(negedge clock);
  endtask
endmodule

interface redmule_reference_if (
    input logic clock,
    reset
);
  function void load_word(int bank, int row, bit [127:0] data);
    tb_top.reference_model.load_word(bank, row, data);
  endfunction
  function bit [127:0] read_word(int row);
    return tb_top.reference_model.memory[2][row];
  endfunction
  function longint get_ticks();
    return tb_top.reference_model.ticks;
  endfunction
  task execute(int m, int k, int base, bit first, output longint elapsed);
    tb_top.reference_model.execute(m, k, base, first, elapsed);
  endtask
endinterface
