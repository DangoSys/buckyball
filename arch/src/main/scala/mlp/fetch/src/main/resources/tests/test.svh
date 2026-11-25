class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  virtual fetch_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual fetch_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Fetch timing interface is required")
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
  driver control;
  invalid_peer peer;
  program_image image;
  reset_controller reset;
  uvm_sequencer #(control_item) seqr;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    image = new;
    control = driver::type_id::create("control", this);
    peer = invalid_peer::type_id::create("peer", this);
    reset = reset_controller::type_id::create("reset", this);
    seqr = new("seqr", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    control.seq_item_port.connect(seqr.seq_item_export);
    control.image = image;
    peer.image = image;
  endfunction
  task execute();
    invalid_sequence seq = invalid_sequence::type_id::create("violation");
    seq.reset = reset;
    seq.seqr  = seqr;
    seq.start(null);
  endtask
endclass
