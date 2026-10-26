class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  virtual chi_retry_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    if (`CHI_NODE_BITS != 7 || `CHI_ADDRESS_BITS != 44 || `CHI_NODE_ID != 1)
      `uvm_fatal("PROFILE",
                 "CHI packet scenarios require 7-bit nodes, 44-bit addresses and requester 1")
    if (`CHI_RECORDS != 4) `uvm_fatal("PROFILE", "Retry saturation scenarios require four records")
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual chi_retry_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI timing interface is required")
  endfunction
  task execute();
    scenario_sequence seq = scenario_sequence::type_id::create("scenario");
    seq.test_env = test_env;
    seq.state = test_env.state;
    seq.vif = vif;
    seq.scenario = test_env.state.scenario;
    seq.start(null);
  endtask
endclass
