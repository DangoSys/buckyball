class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  virtual port_if request;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual port_if)::get(this, "", "request", request))
      `uvm_fatal("VIF", "Bank test clock interface missing")
  endfunction

  task execute();
    access_sequence seq = access_sequence::type_id::create("seq");
    seq.start(test_env.bus.seqr);
    test_env.scoreboard.wait_checked(seq.expected_responses);
    // Observe the next idle cycles before ending the test.
    repeat (2) @(negedge request.clock);
    if (test_env.scoreboard.checked != seq.expected_responses ||
        test_env.bus.mon.retired != seq.expected_responses ||
        test_env.bus.mon.cancelled != seq.expected_cancellations ||
        test_env.bus.mon.accepted != seq.expected_responses + seq.expected_cancellations)
      `uvm_fatal("BANK_COMPLETE", "Bank transaction accounting mismatch")
    `uvm_info("BANK", $sformatf("accepted=%0d checked=%0d cancelled=%0d pending=%0d",
                                test_env.bus.mon.accepted, test_env.scoreboard.checked,
                                test_env.bus.mon.cancelled, test_env.scoreboard.pending()), UVM_LOW)
  endtask
endclass
