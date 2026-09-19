class env extends uvm_env;
  `uvm_component_utils(env)

  driver source;
  checker monitor;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    source = driver::type_id::create("source", this);
    monitor = checker::type_id::create("monitor", this);
  endfunction
endclass

class protocol_test extends uvm_test;
  `uvm_component_utils(protocol_test)

  env test_env;
  virtual arbiter_if vif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual arbiter_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("VIF", "test requires vif")
    end
    test_env = env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    wait (test_env.source.done);
    repeat (2) @(posedge vif.clock);
    phase.drop_objection(this);
  endtask
endclass
