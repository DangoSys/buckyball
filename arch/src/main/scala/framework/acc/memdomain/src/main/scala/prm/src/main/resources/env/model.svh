class memory_model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(memory_model)
  logic [WIDTH-1:0] words[BANKS][ROWS];
  bit [MASKS-1:0] initialized[BANKS][ROWS];
  int expected_serial[BANKS][2], actual_serial[BANKS][2];
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  covergroup accesses with function sample (int bank, bit write, int row, logic [MASKS-1:0] mask);
    option.per_instance = 1;
    bank_index: coverpoint bank {bins banks[] = {[0 : BANKS - 1]};}
    direction: coverpoint write;
    address: coverpoint row {
      bins first = {0}; bins interior = {[1 : ROWS - 2]}; bins last = {ROWS - 1};
    }
    write_mask: coverpoint mask iff (write) {
      bins none = {'0}; bins full = {'1}; bins partial = default;
    }
    bank_direction: cross bank_index, direction;
  endgroup
  covergroup protocol with function sample (
      bit clear, bit clearing, bit read_stall, bit write_stall, bit collision
  );
    option.per_instance = 1;
    clear_request: coverpoint clear;
    clearing_active: coverpoint clearing;
    read_backpressure: coverpoint read_stall;
    write_backpressure: coverpoint write_stall;
    read_write_collision: coverpoint collision;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    reset_ap = new("reset_ap", this);
    accesses = new();
    protocol = new();
    foreach (expected_serial[b, d]) begin
      expected_serial[b][d] = 0;
      actual_serial[b][d]   = 0;
    end
  endfunction
  function void write(cycle_item t);
    result_item expected, actual;
    if (t.reset) begin
      reset_ap.write(1);
      foreach (actual_serial[b, d]) actual_serial[b][d] = expected_serial[b][d];
      return;
    end
    foreach (t.sampled[b]) begin
      bank_signals s = t.sampled[b];
      protocol.sample(s.clear, s.clearing, s.read_result_valid && !s.read_result_ready,
                      s.write_result_valid && !s.write_result_ready, s.read_valid && s.write_valid);
      if (s.clear || s.clearing)
        evidence::check(!s.read_ready && !s.write_ready, "PRM accepted an access during clear");
      if (s.clear) begin
        for (int row = 0; row < ROWS; row++) begin
          words[b][row] = '0;
          initialized[b][row] = '1;
        end
      end
      if (s.write_valid && s.write_ready) begin
        evidence::check(!s.read_ready, "PRM read bypassed a same-cycle write");
        for (int m = 0; m < MASKS; m++)
        if (s.write_mask[m]) begin
          words[b][s.write_addr][m*(WIDTH/MASKS)+:(WIDTH/MASKS)] = s.write_data[m*(WIDTH/MASKS)+:(WIDTH/MASKS)];
          initialized[b][s.write_addr][m] = 1;
        end
        expected = result_item::type_id::create("write_ack");
        expected.bank = b;
        expected.write = 1;
        expected.data = 0;
        expected.ok = 1;
        expected.set_transaction_id((expected_serial[b][1]++ * BANKS + b) * 2 + 1);
        expected_ap.write(expected);
        accesses.sample(b, 1, s.write_addr, s.write_mask);
      end
      if (s.read_valid && s.read_ready) begin
        evidence::check(initialized[b][s.read_addr] == '1, "PRM read of uninitialized storage");
        expected = result_item::type_id::create("read");
        expected.bank = b;
        expected.write = 0;
        expected.data = words[b][s.read_addr];
        expected.ok = 1;
        expected.set_transaction_id((expected_serial[b][0]++ * BANKS + b) * 2);
        expected_ap.write(expected);
        accesses.sample(b, 0, s.read_addr, '0);
      end
      if (s.write_result_valid && s.write_result_ready) begin
        actual = result_item::type_id::create("write_result");
        actual.bank = b;
        actual.write = 1;
        actual.data = 0;
        actual.ok = s.write_ok;
        actual.set_transaction_id((actual_serial[b][1]++ * BANKS + b) * 2 + 1);
        actual_ap.write(actual);
      end
      if (s.read_result_valid && s.read_result_ready) begin
        actual = result_item::type_id::create("read_result");
        actual.bank = b;
        actual.write = 0;
        actual.data = s.read_data;
        actual.ok = 1;
        actual.set_transaction_id((actual_serial[b][0]++ * BANKS + b) * 2);
        actual_ap.write(actual);
      end
    end
  endfunction
endclass
