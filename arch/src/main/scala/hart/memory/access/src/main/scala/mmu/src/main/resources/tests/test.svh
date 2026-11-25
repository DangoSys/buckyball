class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  virtual mmu_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual mmu_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Walker timing interface is required")
  endfunction
  task execute();
    scenario_sequence seq = scenario_sequence::type_id::create("scenario");
    seq.test_env = test_env;
    seq.vif = vif;
    seq.start(null);
  endtask
endclass
class invalid_test extends base_test;
  `uvm_component_utils(invalid_test)
  driver source;
  reset_controller reset;
  uvm_sequencer #(request_item) seqr;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    source = driver::type_id::create("source", this);
    reset  = reset_controller::type_id::create("reset", this);
    seqr   = new("seqr", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    source.seq_item_port.connect(seqr.seq_item_export);
  endfunction
  task execute();
    invalid_sequence seq = invalid_sequence::type_id::create("scenario");
    reset.pulse(4);
    seq.start(seqr);
  endtask
endclass
