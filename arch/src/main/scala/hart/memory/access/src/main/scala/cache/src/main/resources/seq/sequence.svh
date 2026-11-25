class cache_sequence extends scenario_sequence;
  `uvm_object_utils(cache_sequence)
  virtual cmo_if cmo;
  cache_model checked_state;
  function new(string name = "scenario");
    super.new(name);
  endfunction
  task operation(int op, int lines, int tag, int failure = -1);
    longint unsigned checked_before = checked_state.acknowledgments.checked;
    `uvm_info("CACHE_CMO",
              $sformatf("op=%0d lines=%0d tag=%0d error=%0d", op, lines, tag, failure), UVM_HIGH)
    checked_state.fail_index = failure;
    @(negedge vif.clock);
    cmo.valid = 1;
    cmo.first_line = 'h40000;
    cmo.last_line = 'h40000 + (lines - 1) * 64;
    cmo.operation = op;
    cmo.tag = tag;
    cmo.response_ready = 0;
    do @(posedge vif.clock); while (!cmo.ready);
    @(negedge vif.clock);
    cmo.valid = 0;
    wait (cmo.response_valid);
    repeat (4) @(negedge vif.clock);
    cmo.response_ready = 1;
    wait (checked_state.acknowledgments.checked == checked_before + 1);
    @(negedge vif.clock);
  endtask
  task body();
    `uvm_info("CACHE_STAGE", "Starting demand regression", UVM_HIGH)
    super.body();
    `uvm_info("CACHE_STAGE", "Starting maintenance regression", UVM_HIGH)
    for (int op = 0; op < 3; op++) begin
      cmo.older_drained = 0;
      fork
        operation(op, 1 + op, 20 + op);
        begin
          repeat (8) @(negedge vif.clock);
          evidence::check(!cmo.ready, "Maintenance ignored an older undrained request");
          cmo.older_drained = 1;
        end
      join
      operation(op, 3, 40 + op, op);
    end
    // An independent demand request progresses while a serialized CMO is outstanding.
    state.hold_req = 1;
    fork
      operation(0, 4, 60);
      begin
        repeat (6) @(negedge vif.clock);
        issue(16320);
        state.hold_req = 0;
        settle();
      end
    join
    state.hold_req = 1;
    @(negedge vif.clock);
    cmo.valid = 1;
    cmo.first_line = 'h50000;
    cmo.last_line = 'h50100;
    cmo.operation = 2;
    cmo.tag = 200;
    do @(posedge vif.clock); while (!cmo.ready);
    @(negedge vif.clock);
    cmo.valid = 0;
    repeat (3) @(negedge vif.clock);
    test_env.reset.pulse(3);
    state.hold_req = 0;
    repeat (4) @(negedge vif.clock);
    operation(2, 2, 201);
  endtask
endclass
