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
    seq.test_env = test_env;
    seq.start(test_env.bus.seqr);
  endtask
endclass
class invalid_test extends base_test;
  `uvm_component_utils(invalid_test)
  agent bus;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    bus = agent::type_id::create("bus", this);
  endfunction
  task execute();
    invalid_sequence seq = invalid_sequence::type_id::create("violation");
    seq.start(bus.seqr);
  endtask
endclass
