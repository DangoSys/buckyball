class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env bench;
  reset_controller resetter;
  virtual port_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    resetter = reset_controller::type_id::create("reset", this);
    uvm_config_db#(reset_controller)::set(this, "env.agent.driver", "resetter", resetter);
    bench = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual port_if)::get(this, "", "vif", vif))
      `uvm_fatal("INTERFACE", "Test interface is required")
  endfunction
  task execute();
    stimulus sequence_ = stimulus::type_id::create("sequence");
    resetter.pulse(4);
    sequence_.start(bench.bus.seqr);
    bench.scoreboard.wait_checked(sequence_.sent);
    @(negedge vif.clock);
    evidence::check(evidence::cancelled == 2,
                    "Reset did not cancel both outstanding SPM responses");
    evidence::check(bench.scoreboard.checked == sequence_.sent, "Comparison count mismatch");
  endtask
endclass
