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
    seq.start(test_env.seqr);
  endtask
endclass
class invalid_test extends base_test;
  `uvm_component_utils(invalid_test)
  driver drv;
  uvm_sequencer #(cycle_item) seqr;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    drv  = driver::type_id::create("driver", this);
    seqr = new("seqr", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    drv.seq_item_port.connect(seqr.seq_item_export);
  endfunction
  task execute();
    invalid_sequence seq = invalid_sequence::type_id::create("scenario");
    seq.start(seqr);
  endtask
endclass
