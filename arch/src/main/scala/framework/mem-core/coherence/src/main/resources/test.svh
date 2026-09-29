class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  env test_env;
  virtual coherence_control_if control;
  virtual stream_if #(`COH_REQ_WIDTH) req;
  int serial = 1;
  int reused_txn;
  int expected = 0;
  longint unsigned eviction_addresses[`COH_MSHRS];
  function new(string name, uvm_component parent);
    super.new(name, parent);
    timeout = 200us;
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    test_env = env::type_id::create("env", this);
    if (!uvm_config_db#(virtual coherence_control_if)::get(
            this, "", "control", control
        ) || !uvm_config_db#(virtual stream_if #(`COH_REQ_WIDTH))::get(
            this, "", "req", req
        ))
      `uvm_fatal("VIF", "Coherence test interfaces missing")
    if (`COH_AGENTS < 2 || `COH_SETS < `COH_MSHRS || `COH_WAYS != 2)
      `uvm_fatal("CONFIG",
                 "This regression requires two CPUs, two ways, and at least one set per MSHR")
  endfunction
  function longint unsigned addr(int tag, int set_index);
    return (longint'(tag) * `COH_SETS + set_index) * 64;
  endfunction
  function bit [511:0] pattern(int seed);
    bit [511:0] value;
    for (int word = 0; word < 16; word++) value[word*32+:32] = seed ^ (32'h01010101 * word);
    return value;
  endfunction
  task send(int node, int opcode, longint unsigned address, bit error = 0);
    req_t packet = '0;
    int txn = serial & 4095;
    bit read = opcode == `COH_READ_SHARED || opcode == `COH_READ_NSD || opcode == `COH_READ_UNIQUE;
    packet[`CF(REQ, SRCID)] = node;
    packet[`CF(REQ, TGTID)] = `COH_HOME;
    packet[`CF(REQ, TXNID)] = txn;
    packet[`CF(REQ, OPCODE)] = opcode;
    packet[`CF(REQ, ADDR)] = address;
    packet[`CF(REQ, SIZE)] = 6;
    packet[`CF(REQ, EXPCOMPACK)] = read;
    packet[`CF(REQ, SNPATTR)] = 1;
    packet[`CF(REQ, MEMATTR)] = 12;
    packet[`CF(REQ, ALLOWRETRY)] = 1;
    test_env.expected_error[(node<<12)|txn] = error;
    req.bits  <= packet;
    req.valid <= 1;
    do @(req.sample); while (!req.sample.ready);
    req.valid <= 0;
    serial++;
    expected++;
  endtask
  task settle(int occupied = 0);
    test_env.scoreboard.wait_checked(expected);
    do @(control.cb); while (control.cb.outstanding != occupied);
  endtask
  task execute();
    control.reset = 1;
    req.valid = 0;
    req.bits = '0;
    repeat (4) @(control.cb);
    control.cb.reset <= 0;
    @(control.cb);
    for (int slot = 0; slot < `COH_MSHRS; slot++) begin
      longint unsigned a = addr(8 * slot, slot);
      longint unsigned b = addr(8 * slot + 1, slot);
      longint unsigned c = addr(8 * slot + 2, slot);
      int saved;
      test_env.hold_ack = (1 << slot) - 1;
      for (int h = 0; h < slot; h++) send(1, `COH_READ_SHARED, addr(100 + slot, h));
      settle(slot);
      send(1, `COH_READ_SHARED, a);
      settle(slot);
      send(2, `COH_READ_NSD, a);
      settle(slot);
      test_env.cpus[1][a].valid = 0;
      send(1, `COH_READ_UNIQUE, a);
      settle(slot);
      send(2, `COH_READ_SHARED, a);
      settle(slot);
      send(1, `COH_READ_UNIQUE, a);
      settle(slot);
      test_env.cpu_write(1, a, '1);
      send(2, `COH_READ_SHARED, a);
      send(1, `COH_WRITEBACK, a);
      settle(slot);
      send(1, `COH_READ_UNIQUE, a);
      send(2, `COH_EVICT, a);
      settle(slot);
      test_env.cpu_write(1, a, pattern(32'h12345678));
      send(2, `COH_READ_SHARED, a);
      send(1, `COH_WRITEBACK, a);
      settle(slot);
      send(1, `COH_READ_SHARED, b);
      settle(slot);
      send(2, `COH_READ_SHARED, c);
      settle(slot);
      if (test_env.cpus[1][a].valid)
        `uvm_fatal("INCLUSION", "inclusive victim survived replacement")
      send(1, `COH_READ_SHARED, a);
      settle(slot);
      send(2, `COH_CLEAN_INVALID, a);
      settle(slot);
      send(1, `COH_CLEAN_INVALID, a);
      settle(slot);
      send(1, `COH_EVICT, a);
      settle(slot);
      send(1, `COH_WRITEBACK, a);
      settle(slot);
      send(1, `COH_READ_UNIQUE, a);
      settle(slot);
      test_env.cpu_write(1, a, pattern(32'h87654321));
      send(2, `COH_CLEAN_INVALID, a);
      settle(slot);
      send(2, `COH_READ_SHARED, a);
      settle(slot);
      test_env.fault_addr = addr(8 * slot + 3, slot);
      test_env.fault_enabled = 1;
      send(1, `COH_READ_SHARED, test_env.fault_addr, 1);
      settle(slot);
      test_env.fault_enabled = 0;
      send(1, `COH_READ_SHARED, test_env.fault_addr);
      settle(slot);
      begin
        longint unsigned high = (((64'h1 << `COH_REQ_ADDR_WIDTH) - 1) &
          ~(longint'(`COH_SETS * 64) - 1)) | (slot * 64);
        send(1, `COH_READ_UNIQUE, high);
        settle(slot);
        send(1, `COH_EVICT, high);
        settle(slot);
        send(1, `COH_READ_UNIQUE, high);
        settle(slot);
        test_env.cpu_write(1, high, '0);
        send(1, `COH_WRITEBACK, high);
        settle(slot);
        send(2, `COH_CLEAN_INVALID, high);
        settle(slot);
        send(2, `COH_READ_SHARED, high);
        settle(slot);
        send(1, `COH_CLEAN_INVALID, high);
        settle(slot);
        send(1, `COH_READ_SHARED, addr(8 * slot + 7, slot));
        settle(slot);
      end
      saved  = serial;
      serial = 4095;
      send(2, `COH_READ_SHARED, a);
      settle(slot);
      send(2, `COH_READ_SHARED, a);
      settle(slot);
      serial = saved;
      test_env.hold_ack = 0;
      settle();
    end

    test_env.hold_ack = '1;
    send(1, `COH_READ_SHARED, addr(200, 0));
    test_env.scoreboard.wait_checked(expected);
    fork
      begin
        send(2, `COH_READ_SHARED, addr(201, 0));
      end
      begin
        repeat (4) @(control.cb);
        if (req.ready !== 0 || control.cb.outstanding != 1)
          `uvm_fatal("LOCK", "same-set request entered before CompAck")
        test_env.hold_ack = 0;
      end
    join
    settle();
    test_env.hold_ack = '1;
    reused_txn = serial;
    send(1, `COH_READ_SHARED, addr(202, 0));
    test_env.scoreboard.wait_checked(expected);
    serial = reused_txn;
    fork
      begin
        send(1, `COH_READ_SHARED, addr(203, 1));
      end
      begin
        repeat (4) @(control.cb);
        if (req.ready !== 0 || control.cb.outstanding != 1)
          `uvm_fatal("ID_LIFETIME", "live requester TxnID was reused before retirement")
        test_env.hold_ack = 0;
      end
    join
    settle();
    for (int rotation = 0; rotation < `COH_SETS; rotation++) begin
      for (int pass = 0; pass < `COH_WAYS; pass++) begin
        test_env.hold_memory = 1;
        test_env.hold_mem_req = 1;
        test_env.reverse_memory = 1;
        test_env.saw_reorder = 0;
        for (int slot = 0; slot < `COH_MSHRS; slot++) begin
          eviction_addresses[slot] =
              addr(300 + rotation * `COH_WAYS + pass, (slot + rotation) % `COH_SETS);
          send((slot % 2) + 1, `COH_READ_UNIQUE, eviction_addresses[slot]);
        end
        repeat (20) @(control.cb);
        if (control.cb.outstanding != `COH_MSHRS)
          `uvm_fatal("PARALLEL", "different sets did not occupy independent MSHRs")
        test_env.hold_mem_req = 0;
        repeat (16) @(control.cb);
        test_env.hold_memory = 0;
        settle();
        if (`COH_MSHRS > 1 && !test_env.saw_reorder)
          `uvm_fatal("ORDER", "out-of-order response scenario did not execute")
      end
    end
    for (int slot = 0; slot < `COH_MSHRS; slot++)
      send((slot % 2) + 1, `COH_EVICT, eviction_addresses[slot]);
    settle();
    // Delayed eviction notifications must not remove a newer owner's directory state.
    for (int slot = 0; slot < `COH_MSHRS; slot++)
      send((slot % 2) + 1, `COH_EVICT, addr(
           8 * ((slot + `COH_SETS - 1) % `COH_SETS), (slot + `COH_SETS - 1) % `COH_SETS));
    settle();
    for (int slot = 0; slot < `COH_MSHRS; slot++) begin
      longint unsigned previous_line = eviction_addresses[slot] - `COH_SETS * 64;
      test_env.cpu_write((slot % 2) + 1, previous_line, {16{32'h01020300 + slot}});
      send(((slot + 1) % 2) + 1, `COH_READ_SHARED, previous_line);
    end
    settle();
    for (int slot = 0; slot < `COH_MSHRS; slot++)
      send(1, `COH_CLEAN_INVALID, eviction_addresses[slot] - `COH_SETS * 64);
    settle();

    // Fill the completion queue before each selected worker reaches completion.
    for (int slot = 0; slot < `COH_MSHRS - 1; slot++) begin
      longint unsigned slow = addr(900 + slot, slot);
      send(1, `COH_READ_UNIQUE, slow);
      settle();
      test_env.cpu_write(1, slow, pattern(32'h76543210));
      test_env.hold_ack = (1 << slot) - 1;
      for (int h = 0; h < slot; h++) send(1, `COH_READ_SHARED, addr(950 + slot, h));
      settle(slot);
      test_env.hold_memory = 1;
      test_env.hold_rsp = 1;
      send(2, `COH_CLEAN_INVALID, slow);
      send(2, `COH_EVICT, addr(8 * (`COH_SETS - 1), `COH_SETS - 1));
      send(2, `COH_EVICT, addr(8 * (`COH_SETS - 1), `COH_SETS - 1));
      repeat (20) @(control.cb);
      test_env.hold_memory = 0;
      repeat (20) @(control.cb);
      test_env.hold_rsp = 0;
      settle(slot);
      test_env.hold_ack = 0;
      settle();
    end

    // Each victim has two sharers, so a worker can wait to issue its second snoop.
    for (int s = 0; s < `COH_MSHRS; s++) begin
      send(1, `COH_READ_SHARED, addr(700, s));
      settle();
      send(2, `COH_READ_SHARED, addr(700, s));
      settle();
      send(1, `COH_READ_SHARED, addr(701, s));
      settle();
    end
    begin
      int before_snoops = test_env.snoops;
      fork
        begin
          for (int s = 0; s < `COH_MSHRS; s++) send(1, `COH_READ_SHARED, addr(702, s));
        end
        begin
          wait (test_env.snoops > before_snoops);
          test_env.hold_snp = 1;
          repeat (12) @(control.cb);
          test_env.hold_snp = 0;
        end
      join
    end
    settle();
    // Align a returned memory operation with a new lookup to exercise every Cache arbiter input.
    for (int slot = 0; slot < `COH_MSHRS; slot++) begin
      int other;
      test_env.hold_ack = (1 << slot) - 1;
      for (int h = 0; h < slot; h++) send(1, `COH_READ_SHARED, addr(960 + slot, h));
      settle(slot);
      test_env.hold_memory = 1;
      send(1, `COH_READ_SHARED, addr(980 + slot, slot));
      wait (test_env.mem_queue.size() != 0);
      if (slot == `COH_MSHRS - 1) begin
        test_env.hold_ack[0] = 0;
        do @(control.cb); while (control.cb.outstanding != slot);
        other = 0;
      end else other = slot + 1;
      test_env.hold_memory = 0;
      @(control.cb);
      send(2, `COH_READ_SHARED, addr(990 + slot, other));
      settle($countones(test_env.hold_ack));
      test_env.hold_ack = 0;
      settle();
    end

    // A recalled dirty line reaches memory after two independent reads fill the request queue.
    for (int slot = 0; slot < 2; slot++) begin
      longint unsigned slow = addr(1100 + slot, slot);
      control.cb.reset <= 1;
      repeat (2) @(control.cb);
      control.cb.reset <= 0;
      @(control.cb);
      send(1, `COH_READ_UNIQUE, slow);
      settle();
      test_env.cpu_write(1, slow, pattern(32'hFEDCBA98));
      test_env.hold_ack = (1 << slot) - 1;
      for (int h = 0; h < slot; h++) send(1, `COH_READ_SHARED, addr(1150 + slot, h));
      settle(slot);
      test_env.hold_snp = 1;
      test_env.hold_mem_req = 1;
      send(2, `COH_CLEAN_INVALID, slow);
      send(1, `COH_READ_SHARED, addr(1200 + slot, slot + 1));
      send(2, `COH_READ_SHARED, addr(1250 + slot, slot + 2));
      repeat (16) @(control.cb);
      test_env.hold_snp = 0;
      repeat (16) @(control.cb);
      test_env.hold_mem_req = 0;
      settle(slot);
      test_env.hold_ack = 0;
      settle();
    end
    if (test_env.memory_writes == 0) `uvm_fatal("WRITEBACK", "no dirty victim reached memory")
    control.cb.reset <= 1;
    repeat (2) @(control.cb);
    control.cb.reset <= 0;
    @(control.cb);
    send(1, `COH_READ_SHARED, addr(0, 0));
    settle();
    `uvm_info("COHERENCE", $sformatf("Checked %0d completions, peak MSHRs=%0d, dirty writes=%0d",
                                     test_env.scoreboard.checked, test_env.peak,
                                     test_env.memory_writes), UVM_LOW)
  endtask
endclass
