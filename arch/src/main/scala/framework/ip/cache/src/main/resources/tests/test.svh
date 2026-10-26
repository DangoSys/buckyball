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
    access_sequence seq = access_sequence::type_id::create("scenario");
    seq.test_env = test_env;
    seq.start(test_env.bus.seqr);
    evidence::check(test_env.bus.mon.accepted == test_env.scoreboard.checked + evidence::cancelled,
                    "Cache transaction accounting mismatch");
  endtask
endclass
