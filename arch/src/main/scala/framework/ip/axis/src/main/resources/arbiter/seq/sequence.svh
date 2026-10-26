class protocol_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(protocol_sequence)
  cycle_item signals;
  int unsigned cycles = 0;
  int unsigned transfers = 0;

  function new(string name = "protocol_sequence");
    super.new(name);
  endfunction

  task tick(int unsigned count, int unsigned beats = 0);
    cycle_item req;
    repeat (count) begin
      req = cycle_item::type_id::create("cycle");
      start_item(req);
      req.copy(signals);
      finish_item(req);
      if (!signals.reset) cycles++;
      transfers += beats;
    end
  endtask

  task body();
    signals = cycle_item::type_id::create("signals");
    signals.reset = 1;
    signals.valid = 0;
    signals.data = '{0, 32'h55555555};
    signals.keep = '{4'h1, 4'hf};
    signals.last = 0;
    signals.ready = 0;
    tick(4);
    signals.reset = 0;
    tick(2);

    `uvm_info("SCENARIO", "Simultaneous packets use fixed priority and preserve packet ownership",
              UVM_LOW)
    signals.valid = 3;
    signals.ready = 0;
    tick(2);
    signals.ready = 1;
    tick(1, 1);
    signals.data[0] = '1;
    signals.keep[0] = 15;
    signals.last = 1;
    tick(1, 1);
    signals.valid = 2;
    tick(1, 1);
    signals.data[1] = '1;
    signals.keep[1] = 10;
    signals.last = 2;
    tick(1, 1);
    signals.valid = 0;
    tick(1);

    `uvm_info("SCENARIO", "A late higher-priority input cannot replace a stalled first beat",
              UVM_LOW)
    signals.valid = 2;
    signals.data[1] = 32'hbbbbbbbb;
    signals.keep[1] = 5;
    signals.last = 2;
    signals.ready = 0;
    tick(1);
    signals.valid = 3;
    signals.data[0] = 32'haaaaaaaa;
    signals.keep[0] = 10;
    signals.last = 3;
    tick(2);
    signals.ready = 1;
    tick(1, 1);
    signals.valid = 1;
    tick(1, 1);
    signals.valid = 0;
    tick(1);

    `uvm_info("SCENARIO", "A locked source may pause; stalled TLAST retains ownership", UVM_LOW)
    signals.valid = 2;
    signals.data[1] = 32'h12345678;
    signals.last = 0;
    tick(1, 1);
    signals.valid = 1;
    signals.last  = 1;
    tick(2);
    signals.valid = 3;
    signals.data[1] = 32'h87654321;
    signals.last = 3;
    signals.ready = 0;
    tick(2);
    signals.ready = 1;
    tick(1, 1);
    signals.valid = 1;
    tick(1, 1);
    signals.valid = 0;
    tick(1);

    `uvm_info("SCENARIO", "Single-beat packets preserve every TKEEP mask", UVM_LOW)
    signals.valid = 1;
    signals.last  = 1;
    for (int mask = 0; mask < (1 << (`AXIS_DATA_BITS / 8)); mask++) begin
      signals.keep[0] = mask;
      signals.data[0] = 32'h9e3779b9 * (mask + 1);
      tick(1, 1);
    end
    signals.valid = 0;
    tick(1);

    `uvm_info("SCENARIO", "Reset releases packet ownership", UVM_LOW)
    signals.valid = 2;
    signals.last  = 0;
    tick(1, 1);
    signals.valid = 1;
    signals.last  = 1;
    tick(1);
    signals.reset = 1;
    signals.valid = 0;
    tick(2);
    signals.reset = 0;
    signals.valid = 1;
    tick(1, 1);
    signals.valid = 0;
    tick(1);

    `uvm_info("SCENARIO", "Reset cancels a stalled first beat", UVM_LOW)
    signals.data[1] = 0;
    signals.valid = 2;
    signals.last = 2;
    signals.ready = 0;
    tick(1);
    signals.reset = 1;
    signals.valid = 0;
    tick(2);
    signals.reset = 0;
    signals.valid = 1;
    signals.last  = 1;
    signals.ready = 1;
    tick(1, 1);
    signals.valid = 0;
    tick(2);
  endtask
endclass
