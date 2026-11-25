class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  virtual arbiter_if vif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual arbiter_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Arbiter test requires vif")
    test_env = env::type_id::create("env", this);
  endfunction

  task execute();
    protocol_sequence seq = protocol_sequence::type_id::create("seq");
    seq.start(test_env.bus.seqr);
    test_env.scoreboard.wait_checked(seq.cycles);
    // The monitor finishes the last rising-edge sample before accounting is read.
    @(negedge vif.clock);
    if (test_env.scoreboard.checked != seq.cycles ||
        test_env.bus.mon.accepted != seq.transfers ||
        test_env.bus.mon.transferred != seq.transfers ||
        test_env.scoreboard.pending() != 0 ||
        test_env.scoreboard.pending() != 0)
      `uvm_fatal("ARBITER_COMPLETE", "Arbiter transaction accounting mismatch")
    `uvm_info("ARBITER_CHECKED", $sformatf("cycles=%0d accepted=%0d transferred=%0d pending=0",
                                           test_env.scoreboard.checked, test_env.bus.mon.accepted,
                                           test_env.bus.mon.transferred), UVM_LOW)
  endtask
endclass
