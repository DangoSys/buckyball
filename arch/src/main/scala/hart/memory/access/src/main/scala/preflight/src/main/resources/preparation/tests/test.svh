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
    scenario_sequence seq = scenario_sequence::type_id::create("scenario");
    seq.reference = test_env.reference;
    seq.start(test_env.seqr);
  endtask
endclass
