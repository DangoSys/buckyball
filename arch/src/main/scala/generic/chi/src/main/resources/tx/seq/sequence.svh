class traffic extends uvm_sequence #(cycle_item);
  `uvm_object_utils(traffic)
  env test_env;
  link_signals pins = '0, observed;
  bit reset = 1;
  int grants = 0, cycle = 0, sent = 0;
  function new(string name = "traffic");
    super.new(name);
  endfunction
  task tick(int cycles = 1);
    cycle_item req;
    repeat (cycles) begin
      pins.credit = !reset && pins.active && grants > 0 && cycle % 4 != 0;
      if (pins.credit) grants--;
      req = cycle_item::type_id::create("cycle");
      start_item(req);
      req.pins  = pins;
      req.reset = reset;
      finish_item(req);
      observed = req.sampled;
      if (!reset && observed.output_valid) grants++;
      cycle++;
    end
  endtask
  task restart();
    pins   = '0;
    reset  = 1;
    grants = 0;
    tick(4);
    reset = 0;
    pins.active = 1;
    grants = DEPTH;
    do tick(); while (observed.credits != DEPTH);
  endtask
  task send(logic [WIDTH-1:0] data);
    pins.input_valid = 1;
    pins.input_data  = data;
    do tick(); while (!observed.input_ready);
    pins.input_valid = 0;
    sent++;
  endtask
  task drain();
    while (test_env.scoreboard.checked < sent) tick();
  endtask
  task body();
    restart();
    for (int n = 0; n < 256; n++) send(WIDTH'(32'h9e3779b9 * (n + 1)) ^ ((n % 2) ? '1 : '0));
    drain();
    // Reset cancels the accepted flit before its registered physical output is observed.
    pins.input_valid = 1;
    pins.input_data  = '1;
    do tick(); while (!observed.input_ready);
    restart();
    for (int n = 0; n < WIDTH; n++) send(WIDTH'(1) << n);
    drain();
    tick(3);
    evidence::check(evidence::cancelled == 1,
                    "CHI TX reset did not cancel exactly one in-flight flit");
  endtask
endclass
class invalid_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(invalid_sequence)
  function new(string name = "invalid_sequence");
    super.new(name);
  endfunction
  task body();
    cycle_item req;
    for (int n = 0; n < DEPTH + 8; n++) begin
      req = cycle_item::type_id::create("overflow");
      start_item(req);
      req.pins = '0;
      req.reset = n < 4;
      req.pins.active = n >= 4;
      req.pins.credit = n >= 4;
      finish_item(req);
    end
    `uvm_fatal("EXPECTED_ASSERTION", "CHI TX accepted excess credits")
  endtask
endclass
