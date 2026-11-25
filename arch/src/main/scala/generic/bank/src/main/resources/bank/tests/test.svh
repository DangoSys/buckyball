class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  virtual port_if request;
  virtual port_if response;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual port_if)::get(this, "", "request", request))
      `uvm_fatal("VIF", "Bank test clock interface missing")
    if (!uvm_config_db#(virtual port_if)::get(this, "", "response", response))
      `uvm_fatal("VIF", "Bank test response interface missing")
  endfunction

  task execute();
    access_sequence seq = access_sequence::type_id::create("seq");
    seq.start(test_env.bus.seqr);
    test_env.scoreboard.wait_checked(seq.expected_responses);
    // Fill all four response credits, then drain in order under backpressure.
    for (int mode = 0; mode < 2; mode++) begin
      @(negedge request.clock);
      response.ready = 0;
      for (int i = 0; i < 4; i++) begin
        request.valid = 1;
        request.write = !mode;
        request.addr  = i;
        request.data  = 32'h12340000 + i;
        request.mask  = 15;
        request.tag   = 64 + i;
        @(posedge request.clock);
        if (!request.ready)
          `uvm_fatal("BANK_PIPELINE", "Bank did not accept four consecutive requests")
        @(negedge request.clock);
      end
      request.valid = 0;
      repeat (3) begin
        @(negedge request.clock);
        if (request.ready || !response.valid)
          `uvm_fatal("BANK_CREDIT", "Bank did not hold four pending responses")
      end
      response.ready = 1;
      seq.expected_responses += 4;
      test_env.scoreboard.wait_checked(seq.expected_responses);
    end
    // A ready sink permits one SRAM operation per cycle.
    for (int mode = 0; mode < 2; mode++) begin
      for (int i = 0; i < 32; i++) begin
        @(negedge request.clock);
        request.valid = 1;
        request.write = !mode;
        request.addr  = i % `BANK_ROWS;
        request.data  = 32'h87650000 + i;
        request.mask  = 15;
        request.tag   = 128 + i;
        @(posedge request.clock);
        if (!request.ready)
          `uvm_fatal("BANK_THROUGHPUT", "Bank did not sustain one request per cycle")
      end
      @(negedge request.clock);
      request.valid = 0;
      seq.expected_responses += 32;
      test_env.scoreboard.wait_checked(seq.expected_responses);
    end
    `uvm_info("BANK_THROUGHPUT", "32 writes and 32 reads accepted at one request per cycle",
              UVM_LOW)
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
