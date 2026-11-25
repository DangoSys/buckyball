class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  localparam int N = `ROB_ENTRIES;
  logic [`ROB_INPUT_BITS-1:0] pins = '0;
  logic [`ROB_OUTPUT_BITS-1:0] observed;
  bit reset = 1;
  bit [N-1:0] live = 0, issued = 0, done = 0;
  int ready_offsets[N] = `ROB_READY_OFFSETS;
  function new(string name = "scenario_sequence");
    super.new(name);
  endfunction
  task tick(int cycles = 1);
    cycle_item req;
    int slot;
    repeat (cycles) begin
      req = cycle_item::type_id::create("cycle");
      start_item(req);
      req.inputs = pins;
      req.reset  = reset;
      finish_item(req);
      observed = req.outputs;
      if (reset) begin
        live   = 0;
        issued = 0;
        done   = 0;
      end else begin
        if (`RI(pins, ALLOC_VALID) && `RO(observed, ALLOC_READY)) begin
          slot = `RO(observed, TAIL);
          live[slot] = 1;
          issued[slot] = 0;
          done[slot] = 0;
          `RI(pins, ALLOC_VALID) = 0;
        end
        if (`RI(pins, ISSUE_VALID)) issued[`RI(pins, ISSUE_BITS)] = 1;
        if (`RI(pins, COMPLETE_VALID)) done[`RI(pins, COMPLETE_BITS)] = 1;
        live &= ~`RO(observed, RETIRED);
        issued &= ~`RO(observed, RETIRED);
        done &= ~`RO(observed, RETIRED);
      end
    end
  endtask
  task restart();
    pins  = '0;
    reset = 1;
    tick(2);
    reset = 0;
    tick();
  endtask
  task body();
    int selected;
    restart();
    for (int n = 0; n < N; n++) begin
      `RI(pins, ALLOC_VALID) = 1;
      `RI(pins, ALLOC_BITS_DATA) = 64'hdeadbeef00000000 | n;
      `RI(pins, ALLOC_BITS_DEPENDENCIES) = live;
      `RI(pins, ALLOC_BITS_COMMITBARRIER) = n % 2;
      do tick(); while (`RI(pins, ALLOC_VALID));
    end
    tick(3);
    evidence::check(`RO(observed, FULL), "ROB did not fill");
    for (int slot = 0; slot < N; slot++) begin
      `RI(pins, ISSUE_VALID) = 1;
      `RI(pins, ISSUE_BITS)  = slot;
      tick();
    end
    `RI(pins, ISSUE_VALID) = 0;
    for (int slot = N - 1; slot >= 0; slot--) begin
      `RI(pins, COMPLETE_VALID) = 1;
      `RI(pins, COMPLETE_BITS)  = slot;
      tick();
    end
    `RI(pins, COMPLETE_VALID) = 0;
    tick(2);
    foreach (ready_offsets[i]) pins[ready_offsets[i]] = 1;
    tick(2);
    evidence::check(`RO(observed, EMPTY), "ROB did not retire its completed batch");
    for (int cycle = 0; cycle < 1000; cycle++) begin
      if (!`RI(pins, ALLOC_VALID) && cycle % 3 != 0) begin
        `RI(pins, ALLOC_VALID) = 1;
        `RI(pins, ALLOC_BITS_DATA) = (64'h9e3779b97f4a7c15 * cycle) ^ 64'hcafe00001234beef;
        `RI(pins, ALLOC_BITS_DEPENDENCIES) = live;
        `RI(pins, ALLOC_BITS_COMMITBARRIER) = cycle % 4 == 0;
      end
      `RI(pins, ISSUE_VALID) = 0;
      `RI(pins, COMPLETE_VALID) = 0;
      for (int i = 0; i < N; i++) begin
        selected = (cycle + i) % N;
        if (live[selected] && !issued[selected] && !done[selected]) begin
          `RI(pins, ISSUE_VALID) = 1;
          `RI(pins, ISSUE_BITS)  = selected;
          break;
        end
      end
      for (int i = 0; i < N; i++) begin
        selected = (N - 1 - i + cycle) % N;
        if (live[selected] && issued[selected] && !done[selected]) begin
          `RI(pins, COMPLETE_VALID) = 1;
          `RI(pins, COMPLETE_BITS)  = selected;
          break;
        end
      end
      foreach (ready_offsets[i]) pins[ready_offsets[i]] = (cycle + i) % 5 != 0;
      tick();
    end
    `RI(pins, ALLOC_VALID) = 0;
    `RI(pins, ISSUE_VALID) = 0;
    foreach (ready_offsets[i]) pins[ready_offsets[i]] = 1;
    while (live != 0) begin
      `RI(pins, COMPLETE_VALID) = 0;
      for (int i = 0; i < N; i++)
      if (live[i] && !done[i]) begin
        `RI(pins, COMPLETE_VALID) = 1;
        `RI(pins, COMPLETE_BITS)  = i;
        break;
      end
      tick();
    end
    `RI(pins, COMPLETE_VALID) = 0;
    // A one-cycle unit may issue and complete the same live entry in one clock.
    `RI(pins, ALLOC_VALID) = 1;
    `RI(pins, ALLOC_BITS_DATA) = '1;
    do tick(); while (`RI(pins, ALLOC_VALID));
    selected = `RO(observed, TAIL);
    `RI(pins, ISSUE_VALID) = 1;
    `RI(pins, ISSUE_BITS) = selected;
    `RI(pins, COMPLETE_VALID) = 1;
    `RI(pins, COMPLETE_BITS) = selected;
    tick();
    `RI(pins, ISSUE_VALID) = 0;
    `RI(pins, COMPLETE_VALID) = 0;
    tick(3);
  endtask
endclass
class invalid_sequence extends scenario_sequence;
  `uvm_object_utils(invalid_sequence)
  function new(string name = "invalid_sequence");
    super.new(name);
  endfunction
  task body();
    string scenario;
    if (!$value$plusargs("CASE=%s", scenario))
      `uvm_fatal("CASE", "ROB assertion scenario is required")
    restart();
    if (scenario == "issue") begin
      `RI(pins, ISSUE_VALID) = 1;
      `RI(pins, ISSUE_BITS)  = 0;
    end else begin
      `RI(pins, COMPLETE_VALID) = 1;
      `RI(pins, COMPLETE_BITS)  = 0;
    end
    tick(3);
    `uvm_fatal("EXPECTED_ASSERTION", "Invalid ROB command did not trigger its assertion")
  endtask
endclass
