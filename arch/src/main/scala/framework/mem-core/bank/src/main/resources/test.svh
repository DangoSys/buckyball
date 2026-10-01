class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  virtual bank_if request;
  virtual bank_if response;
  env test_env;
  int checked = 0;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual bank_if)::get(
            this, "", "request", request
        ) || !uvm_config_db#(virtual bank_if)::get(
            this, "", "response", response
        ))
      `uvm_fatal("VIF", "bank interfaces missing")
    test_env = env::type_id::create("env", this);
  endfunction

  task access (bit wr, bit [3:0] addr, bit [31:0] data, bit [3:0] mask, bit [7:0] tag, int stall);
    int unsigned target = test_env.scoreboard.checked + 1;
    @(negedge request.clock);
    request.valid = 1;
    request.write = wr;
    request.addr = addr;
    request.data = data;
    request.mask = mask;
    request.tag = tag;
    response.ready = 0;
    do @(posedge request.clock); while (!request.ready);
    @(negedge request.clock);
    request.valid = 0;
    while (!response.valid) @(negedge request.clock);
    repeat (stall + 1) begin
      if (response.valid !== 1'b1) `uvm_fatal("BANK", "response valid dropped under backpressure")
      if (request.ready !== 1'b0)
        `uvm_fatal("BANK", "accepted another request with an outstanding response")
      @(negedge request.clock);
    end
    response.ready = 1;
    @(negedge request.clock);
    if (response.valid !== 1'b0) `uvm_fatal("BANK", "response was not retired")
    response.ready = 0;
    test_env.scoreboard.wait_checked(target);
    checked++;
  endtask

  task reset_pending(bit wr);
    @(negedge request.clock);
    request.valid = 1;
    request.write = wr;
    request.addr = 15;
    request.data = 'hface1234;
    request.mask = 15;
    request.tag = 255;
    response.ready = 0;
    do @(posedge request.clock); while (!request.ready);
    @(negedge request.clock);
    request.valid  = 0;
    request.reset  = 1;
    response.reset = 1;
    repeat (2) @(negedge request.clock);
    request.reset  = 0;
    response.reset = 0;
    repeat (3) begin
      @(negedge request.clock);
      if (response.valid !== 1'b0 || request.ready !== 1'b1)
        `uvm_fatal("RESET", "reset did not discard outstanding transaction")
    end
    access (0, 15, 0, 0, 42, 3);
  endtask

  task execute();
    request.valid = 0;
    request.write = 0;
    request.addr = 0;
    request.data = 0;
    request.mask = 0;
    request.tag = 0;
    request.error = 0;
    request.reset = 1;
    response.reset = 1;
    response.ready = 0;
    repeat (4) @(negedge request.clock);
    request.reset  = 0;
    response.reset = 0;
    for (int a = 0; a < 16; a++) access (1, a, $urandom(), 15, a, 0);
    for (int a = 0; a < 16; a++) access (0, a, 0, 0, 255 - a, 2);
    for (int m = 0; m < 16; m++) begin
      access (1, m, $urandom(), m, m, 5);
      access (0, m, 0, 0, m, 0);
    end
    reset_pending(1);
    reset_pending(0);
    for (int a = 0; a < 16; a++) access (0, a, 0, 0, a, 0);
    response.ready = 1;
    repeat (2) begin
      @(negedge request.clock);
      if (response.valid !== 1'b0) `uvm_fatal("BANK", "unexpected response while idle")
    end
    `uvm_info("BANK", $sformatf("Checked %0d responses", checked), UVM_LOW)
  endtask
endclass
