class traffic extends uvm_sequence #(cycle_item);
  `uvm_object_utils(traffic)
  env test_env;
  link_signals pins = '0, observed;
  bit reset = 1, vary_ready = 0;
  int budget = 0, cycle = 0, sent = 0;
  function new(string name = "traffic");
    super.new(name);
  endfunction
  task tick(int cycles = 1);
    cycle_item req;
    repeat (cycles) begin
      if (vary_ready) pins.output_ready = cycle % 7 < 3;
      req = cycle_item::type_id::create("cycle");
      start_item(req);
      req.pins  = pins;
      req.reset = reset;
      finish_item(req);
      observed = req.sampled;
      if (!reset) budget += int'(observed.credit) - int'(pins.input_valid);
      cycle++;
    end
  endtask
  task restart();
    pins = '0;
    reset = 1;
    budget = 0;
    vary_ready = 0;
    tick(4);
    reset = 0;
    pins.active = 1;
    pins.pend = 1;
  endtask
  task send(logic [WIDTH-1:0] data);
    while (budget == 0) tick();
    pins.input_valid = 1;
    pins.input_data  = data;
    tick();
    pins.input_valid = 0;
    sent++;
  endtask
  task drain();
    while (test_env.scoreboard.checked < sent) tick();
  endtask
  task body();
    restart();
    for (int n = 0; n < DEPTH; n++) send(WIDTH'(n));
    tick(8);
    evidence::check(budget == 0 && observed.output_valid, "CHI RX did not retain a full buffer");
    pins.output_ready = 1;
    drain();
    vary_ready = 1;
    for (int n = 0; n < 256; n++) send(WIDTH'(32'h9e3779b9 * (n + 1)) ^ ((n % 2) ? '1 : '0));
    pins.output_ready = 1;
    vary_ready = 0;
    drain();
    tick(DEPTH + 2);
    pins.output_ready = 0;
    for (int n = 0; n < DEPTH; n++) send('1);
    tick(3);
    restart();
    sent -= DEPTH;
    pins.output_ready = 1;
    for (int n = 0; n < WIDTH; n++) send(WIDTH'(1) << n);
    drain();
    tick(2);
    evidence::check(evidence::cancelled == DEPTH, "CHI RX reset cancellation count mismatch");
  endtask
endclass
class invalid_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(invalid_sequence)
  function new(string name = "invalid_sequence");
    super.new(name);
  endfunction
  task body();
    cycle_item req;
    for (int n = 0; n < 2 * DEPTH + 8; n++) begin
      req = cycle_item::type_id::create("overflow");
      start_item(req);
      req.pins = '0;
      req.reset = n < 4;
      req.pins.active = n >= 4;
      req.pins.pend = n >= 4;
      req.pins.input_valid = n >= DEPTH + 6;
      finish_item(req);
    end
    `uvm_fatal("EXPECTED_ASSERTION", "CHI RX accepted an ungranted flit")
  endtask
endclass
