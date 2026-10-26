class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
  endfunction
  task execute();
    traffic seq = traffic::type_id::create("traffic");
    test_env.reset.pulse(4);
    seq.start(test_env.seqr);
    test_env.reset.pulse(2);
  endtask
endclass
