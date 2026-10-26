class access_sequence extends uvm_sequence #(request_item);
  `uvm_object_utils(access_sequence)
  env test_env;
  int unsigned expected = 0;
  bit [`CACHE_ID_BITS-1:0] serial = 0;
  bit check_rate = 1, idle_between = 0;
  function new(string name = "access_sequence");
    super.new(name);
  endfunction
  task pulse_reset(int unsigned cycles);
    request_item req = request_item::type_id::create("reset");
    start_item(req);
    req.reset_command = 1;
    req.reset_cycles  = cycles;
    finish_item(req);
  endtask
  function longint unsigned address(int tag, int set_index);
    return (longint'(tag) * `CACHE_SETS + set_index) * `CACHE_LINE_BYTES;
  endfunction
  task send(int op, longint unsigned addr, int way = 0, bit [`CACHE_LINE_BYTES-1:0] mask = '1,
            bit [`CACHE_WAYS-1:0] eligible = '1, bit [`CACHE_LINE_BITS-1:0] data = '0,
            int metadata = 0);
    request_item req = request_item::type_id::create("access");
    start_item(req);
    req.id = serial;
    req.op = op;
    req.addr = addr;
    req.way = way;
    req.data = data;
    req.mask = mask;
    req.metadata = metadata;
    req.eligible = eligible;
    req.check_rate = check_rate;
    req.idle_between = idle_between;
    finish_item(req);
    expected++;
    serial++;
  endtask
  task drain();
    test_env.scoreboard.wait_checked(expected);
    @(test_env.bus.mon.request.cb);
  endtask
  task body();
    pulse_reset(4);
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
    test_env.bus.receiver.blocked = 1;
    repeat (2) @(test_env.bus.mon.request.cb);
    fork
      begin
        for (int i = 0; i < `CACHE_RESPONSE_DEPTH + 2; i++) send(LOOKUP, address(2, 0));
      end
      begin
        repeat (`CACHE_RESPONSE_DEPTH + 5) @(test_env.bus.mon.request.cb);
        test_env.bus.receiver.blocked = 0;
      end
    join
    drain();

    test_env.bus.receiver.blocked = 1;
    repeat (2) @(test_env.bus.mon.request.cb);
    for (int i = 0; i < `CACHE_RESPONSE_DEPTH; i++) send(READ, address(1, 1), 0);
    repeat (2) @(test_env.bus.mon.request.cb);
    pulse_reset(2);
    expected = test_env.scoreboard.checked;
    test_env.bus.receiver.blocked = 0;
    @(test_env.bus.mon.request.cb);
    for (int s = 0; s < `CACHE_SETS; s++) send(LOOKUP, address(1, s));
    drain();
    evidence::check(test_env.bus.mon.retired == expected, "Cache response count mismatch");
  endtask
endclass
