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
    protocol_sequence seq;
    seq = protocol_sequence::type_id::create("seq");
    seq.start(test_env.source.seqr);
    test_env.scoreboard.wait_checked(32);
    `uvm_info("AXIS_CHECKED", $sformatf("Checked %0d transfers", test_env.scoreboard.checked),
              UVM_LOW)
  endtask
endclass
