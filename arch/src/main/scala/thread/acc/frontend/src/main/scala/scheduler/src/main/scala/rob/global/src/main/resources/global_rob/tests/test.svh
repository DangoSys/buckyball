class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  virtual global_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual global_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Global ROB timing interface is required")
    if (`GROB_DIFF || `GROB_GP)
      `uvm_fatal("PROFILE", "Global ROB scenarios require normal execution with private banks")
  endfunction
  task execute();
    scenario_sequence seq = scenario_sequence::type_id::create("scenario");
    seq.test_env = test_env;
    seq.vif = vif;
    seq.start(null);
  endtask
endclass
