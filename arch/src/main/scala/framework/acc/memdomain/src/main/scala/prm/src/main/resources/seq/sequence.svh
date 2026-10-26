class access_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(access_sequence)
  env test_env;
  bank_signals pins[BANKS], observed[BANKS];
  bit reset = 1;
  int checked = 0;
  function new(string name = "access_sequence");
    super.new(name);
  endfunction
  function logic [WIDTH-1:0] pattern(int bank, int seed);
    logic [WIDTH-1:0] value;
    for (int bit_index = 0; bit_index < WIDTH; bit_index++)
    value[bit_index] = ((seed * 32'h9e3779b9) >> (bit_index % 32)) ^ ((bit_index + bank) % 3 == 0);
    return value;
  endfunction
  task tick(int cycles = 1);
    cycle_item req;
    repeat (cycles) begin
      req = cycle_item::type_id::create("cycle");
      start_item(req);
      req.reset = reset;
      req.pins  = pins;
      finish_item(req);
      observed = req.sampled;
    end
  endtask
  task drain(int count);
    checked += count;
    while (test_env.scoreboard.checked != checked) tick();
  endtask
  task access (bit write, int row, logic [MASKS-1:0] mask = '1, int seed = 1, int stall = 0);
    bit pending[BANKS];
    foreach (pins[b]) begin
      pending[b] = 1;
      pins[b].read_valid = !write;
      pins[b].read_addr = row;
      pins[b].write_valid = write;
      pins[b].write_addr = row;
      pins[b].write_data = pattern(b, seed);
      pins[b].write_mask = mask;
      pins[b].read_result_ready = stall == 0;
      pins[b].write_result_ready = stall == 0;
    end
    do begin
      tick();
      foreach (pins[b])
      if (pending[b] && (write ? observed[b].write_ready : observed[b].read_ready)) begin
        pending[b] = 0;
        pins[b].read_valid = 0;
        pins[b].write_valid = 0;
      end
    end while (pending.or());
    tick(stall);
    foreach (pins[b]) begin
      pins[b].read_result_ready  = 1;
      pins[b].write_result_ready = 1;
    end
    drain(BANKS);
  endtask
  task reset_pending(bit write);
    foreach (pins[b]) begin
      pins[b].read_valid = !write;
      pins[b].read_addr = ROWS - 1;
      pins[b].write_valid = write;
      pins[b].write_addr = ROWS - 1;
      pins[b].write_data = pattern(b, 77);
      pins[b].write_mask = '1;
      pins[b].read_result_ready = 0;
      pins[b].write_result_ready = 0;
    end
    tick();
    foreach (pins[b]) begin
      evidence::check(write ? observed[b].write_ready : observed[b].read_ready,
                      "PRM cancellation request was not accepted");
      pins[b].read_valid  = 0;
      pins[b].write_valid = 0;
    end
    tick(3);
    reset = 1;
    tick(2);
    reset = 0;
    foreach (pins[b]) begin
      pins[b].read_result_ready  = 1;
      pins[b].write_result_ready = 1;
    end
    tick();
    access (0, ROWS - 1);
  endtask
  task body();
    foreach (pins[b]) pins[b] = '0;
    tick(4);
    reset = 0;
    foreach (pins[b]) begin
      pins[b].clear = 1;
      pins[b].read_result_ready = 1;
      pins[b].write_result_ready = 1;
    end
    tick();
    foreach (pins[b]) pins[b].clear = 0;
    tick(ROWS + 2);
    for (int row = 0; row < ROWS; row++) access (0, row);
    for (int row = 0; row < ROWS; row++) access (1, row, '1, row + 1);
    for (int row = 0; row < ROWS; row++) access (0, row, '0, 0, 3);
    for (int m = 0; m < MASKS; m++) begin
      access (1, m % ROWS, MASKS'(1) << m, m + 100, 3);
      access (0, m % ROWS);
    end
    access (1, ROWS - 1, '0, 0);
    access (0, ROWS - 1);
    // Concurrent read/write requests use the write port first, then read back its update.
    foreach (pins[b]) begin
      pins[b].read_valid  = 1;
      pins[b].read_addr   = 0;
      pins[b].write_valid = 1;
      pins[b].write_addr  = 0;
      pins[b].write_data  = pattern(b, 999);
      pins[b].write_mask  = '1;
    end
    tick();
    foreach (pins[b]) begin
      evidence::check(observed[b].write_ready && !observed[b].read_ready,
                      "PRM did not prioritize a write");
      pins[b].write_valid = 0;
    end
    tick();
    foreach (pins[b]) begin
      evidence::check(observed[b].read_ready, "PRM read did not resume");
      pins[b].read_valid = 0;
    end
    drain(2 * BANKS);
    reset_pending(0);
    reset_pending(1);
    foreach (pins[b]) pins[b].clear = 1;
    tick();
    foreach (pins[b]) pins[b].clear = 0;
    tick(ROWS + 2);
    for (int row = 0; row < ROWS; row++) access (0, row);
    evidence::check(evidence::cancelled == 2 * BANKS, "PRM reset cancellation count mismatch");
  endtask
endclass
