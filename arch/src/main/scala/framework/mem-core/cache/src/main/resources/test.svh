class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  virtual cache_request_if request;
  env test_env;
  int unsigned expected = 0;
  bit [`CACHE_ID_BITS-1:0] serial = 0;
  bit check_rate = 1;
  bit idle_between = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual cache_request_if)::get(this, "", "request", request))
      `uvm_fatal("VIF", "Cache request interface missing")
  endfunction
  function longint unsigned address(int tag, int set_index);
    return (longint'(tag) * `CACHE_SETS + set_index) * `CACHE_LINE_BYTES;
  endfunction
  task send(int op, longint unsigned addr, int way = 0, bit [`CACHE_LINE_BYTES-1:0] mask = '1,
            bit [`CACHE_WAYS-1:0] eligible = '1, bit [`CACHE_LINE_BITS-1:0] data = '0,
            int metadata = 0);
    int cycles = 0;
    request.cb.valid <= 1;
    request.cb.id <= serial;
    request.cb.op <= op;
    request.cb.addr <= addr;
    request.cb.way <= way;
    request.cb.data <= data;
    request.cb.mask <= mask;
    request.cb.metadata <= metadata;
    request.cb.eligible <= eligible;
    do begin
      @(request.cb);
      cycles++;
    end while (!request.cb.ready);
    if (check_rate && cycles != 1)
      `uvm_fatal("THROUGHPUT", "Cache inserted a request bubble without response backpressure")
    request.cb.valid <= 0;
    expected++;
    serial++;
    if (idle_between) @(request.cb);
  endtask
  task drain();
    test_env.scoreboard.wait_checked(expected);
    @(request.cb);
  endtask
  task execute();
    request.reset = 1;
    request.valid = 0;
    request.id = 0;
    request.op = 0;
    request.addr = 0;
    request.way = 0;
    request.data = 0;
    request.mask = 0;
    request.metadata = 0;
    request.eligible = 0;
    repeat (4) @(request.cb);
    request.cb.reset <= 0;
    @(request.cb);
    send(LOOKUP, 0, 0, '0, '0);
    send(LOOKUP, 0);
    for (int s = 0; s < `CACHE_SETS; s++) begin
      for (int w = 0; w < `CACHE_WAYS; w++) begin
        send(READ, address(w + 1, s), w);
        send(FILL, address(w + 1, s), w, '1, '1, w[0] ? '1 : '0, w);
        send(LOOKUP, address(w + 1, s));
        send(WRITE, address(w + 1, s), w, {(`CACHE_LINE_BYTES / 2) {2'b01}}, '1, w[0] ? '0 : '1,
             '1);
        send(READ, address(w + 1, s), w);
      end
    end
    send(LOOKUP, address(`CACHE_WAYS + 1, 0));
    send(LOOKUP, address(`CACHE_WAYS + 1, 0), 0, '0, '0);
    for (int w = 0; w < `CACHE_WAYS; w++) send(LOOKUP, address(`CACHE_WAYS + 1, 0), 0, '0, 1 << w);
    send(INVALIDATE, address(1, 0), 0);
    send(LOOKUP, address(1, 0));
    send(FILL, address(`CACHE_WAYS + 1, 0), 0, '1, '1, '1, '1);
    send(LOOKUP, address(`CACHE_WAYS + 2, 0));
    send(WRITE, address(`CACHE_WAYS + 1, 0), 0, '0, '1, '0, 0);
    send(READ, address(`CACHE_WAYS + 1, 0), 0);
    send(INVALIDATE, address(`CACHE_WAYS + 1, 0), 0);
    send(INVALIDATE, address(`CACHE_WAYS + 1, 0), 0);
    drain();

    idle_between = 1;
    for (int s = 0; s < `CACHE_SETS; s++) begin
      for (int w = 0; w < `CACHE_WAYS; w++) send(INVALIDATE, address(0, s), w);
      for (int w = 0; w < `CACHE_WAYS; w++) begin
        longint unsigned high_addr = (((64'h1 << `CACHE_ADDR_BITS) - 1) &
          ~(longint'(`CACHE_SETS * `CACHE_LINE_BYTES) - 1)) | (s * `CACHE_LINE_BYTES);
        send(FILL, address(0, s), w, '1, '1, '0, 0);
        send(LOOKUP, address(0, s), 0, '0, '0);
        send(FILL, high_addr, w, '1, '1, '1, '1);
        send(LOOKUP, high_addr);
        send(WRITE, high_addr, w, '1, '1, '0, 0);
        send(READ, high_addr, w);
        send(FILL, address(0, s), w, '1, '1, '0, 0);
        send(READ, address(0, s), w);
        send(INVALIDATE, address(0, s), w);
      end
    end
    drain();

    idle_between = 0;
    serial = '1;
    send(READ, address(0, 0), 0, '0, '0);
    drain();
    serial = 0;
    send(READ, address(0, 0), 0, '0, '0);
    drain();
    check_rate = 0;
    test_env.block_responses = 1;
    repeat (2) @(request.cb);
    fork
      begin
        for (int i = 0; i < `CACHE_RESPONSE_DEPTH + 2; i++) send(LOOKUP, address(2, 0));
      end
      begin
        repeat (`CACHE_RESPONSE_DEPTH + 5) @(request.cb);
        test_env.block_responses = 0;
      end
    join
    drain();

    test_env.block_responses = 1;
    repeat (2) @(request.cb);
    for (int i = 0; i < `CACHE_RESPONSE_DEPTH; i++) send(READ, address(1, 1), 0);
    repeat (2) @(request.cb);
    request.cb.reset <= 1;
    repeat (2) @(request.cb);
    expected = test_env.scoreboard.checked;
    request.cb.reset <= 0;
    test_env.block_responses = 0;
    @(request.cb);
    for (int s = 0; s < `CACHE_SETS; s++) send(LOOKUP, address(1, s));
    drain();
    `uvm_info("CACHE", $sformatf("Checked %0d responses", test_env.scoreboard.checked), UVM_LOW)
  endtask
endclass
