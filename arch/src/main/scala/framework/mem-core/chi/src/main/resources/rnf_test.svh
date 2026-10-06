class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  virtual chi_rnf_if vif;
  chandle model;
  bit [511:0] memory[longint unsigned];
  longint unsigned wb_address[int unsigned], snoop_address;
  bit [511:0] snoop_data;
  bit [1:0] snoop_seen;
  bit [511:0] wb_data[int unsigned];
  bit [1:0] wb_seen[int unsigned];
  bit ack_pending[int unsigned];
  int home_allocations = 0, snoop_allocations = 0;
  int home_ids[$];
  int snoop_id;
  int fill_packets = 0, partial_goal = 0;
  bit hold_dat = 0, forbid_early_snoop = 0, expect_snoop_data = 0;
  bit [63:0] expected[$];
  bit expected_errors[$];
  bit [1:0] fail_read = 0;
  bit mixed_data_error = 0;
  rsp_t rx_rsp_queue[$];
  dat_t rx_dat_queue[$];
  int cycle = 0, submitted = 0, checked = 0, writebacks = 0, snoops = 0, peak = 0;
  bit hold_fill = 0, reverse_transactions = 0, hold_req = 0, hold_rsp = 0, hold_output_dat = 0;
  int cancelled_writebacks = 0, shared_writebacks = 0, cancelled_cpu = 0;
  bit rsp_active = 0, dat_active = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    timeout = 450us;
    if (!uvm_config_db#(virtual chi_rnf_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "RN-F interface missing")
    model = rnf_ref_create(44);
    home_ids.push_back(0);
    home_ids.push_back(4095);
    // Odd multiplication and Gray coding form a permutation of all 12-bit IDs.
    // This avoids correlating a cache bank or response kind with fixed ID bits.
    for (int i = 0; i < 4096; i++) begin
      int encoded = (i * 4051) & 4095;
      int id = encoded ^ (encoded >> 1);
      if (id != 0 && id != 4095) home_ids.push_back(id);
    end
  endfunction
  function int allocate_home_id();
    int id;
    if (home_allocations >= 4096) `uvm_fatal("HOME_ID", "Test exhausted the 12-bit Home ID pool")
    home_allocations++;
    return home_ids.pop_front();
  endfunction
  function void consume_req(req_t r);
    rsp_t response = '0;
    dat_t d = '0;
    if (r.src != 1 || r.tgt != 64 || r.txn >= 2 || r.size != 6 || r.addr[5:0] != 0 || !r.allow_retry)
      `uvm_fatal("REQ", "RN-F request outside declared no-retry Home contract")
    response.src = 64;
    response.tgt = 1;
    response.txn = r.txn;
    if (r.opcode == 'h26 || r.opcode == 'h07) begin
      if (!r.exp_comp_ack || !memory.exists(r.addr))
        `uvm_fatal("READ", "Uninitialized read or missing CompAck")
      d.src = 64;
      d.tgt = 1;
      d.home = 64;
      d.txn = r.txn;
      d.dbid = allocate_home_id();
      ack_pending[d.dbid] = 1;
`ifdef RNF_BAD_DBID_WIDTH
      d.dbid = 'h1000;
`endif
      // B9.1.3 permits any Resp for NDERR; both packets must retain it.
      d.opcode = 4;
      d.error = fail_read;
      d.resp = fail_read == 3 ? 7 : r.opcode == 'h07 ? 2 : 1;
      d.be = '1;
      // Return the upper 256b first; DataID units are 128b.
      d.data_id = 2;
      d.data = fail_read ? 0 : memory[r.addr][511:256];
      rx_dat_queue.push_back(d);
      d.data_id = 0;
      d.data = fail_read ? 0 : memory[r.addr][255:0];
      if (mixed_data_error) d.error = 0;  // first packet error must survive a clean final packet
`ifdef RNF_BAD_DBID
      d.dbid = d.dbid ^ 16'h0001;
`endif
`ifdef RNF_BAD_RESP
      d.resp = 2;
`endif
      rx_dat_queue.push_back(d);
    end else if (r.opcode == 'h1b) begin
      response.dbid   = allocate_home_id();
      response.opcode = 5;
      rx_rsp_queue.push_back(response);
      wb_address[response.dbid] = r.addr;
      wb_seen[response.dbid] = 0;
    end else if (r.opcode == 'h0d) begin
      response.opcode = 4;
      rx_rsp_queue.push_back(response);
    end else `uvm_fatal("REQ", "Unexpected request opcode")
  endfunction
  function void consume_data(dat_t d);
    bit [511:0] golden;
    int index;
    if (d.tgt != 64 || d.src != 1 || !(d.data_id == 0 || d.data_id == 2))
      `uvm_fatal("DAT", "Unexpected RN-F data endpoint/DataID")
    index = d.data_id / 2;
    if (d.opcode == 2) begin
      if (d.resp == 0 && (d.be != 0 || d.data != 0))
        `uvm_fatal("WB_ZERO_DATA", "CopyBack_I must carry zero data and zero byte enables")
      if (!wb_address.exists(d.txn) || wb_seen[d.txn][index])
        `uvm_fatal("WB", "Unknown/duplicate writeback")
      wb_seen[d.txn][index] = 1;
      wb_data[d.txn][index*256+:256] = d.data;
      if (!(&d.be) && d.resp != 0) `uvm_fatal("WB", "Valid writeback has empty BE")
      if (&wb_seen[d.txn]) begin
        if (d.resp == 0) begin
          if (d.be != 0) `uvm_fatal("WB", "Cancelled writeback carries valid bytes")
          cancelled_writebacks++;
        end else begin
          if (rnf_ref_read(model, wb_address[d.txn], golden) || wb_data[d.txn] !== golden)
            `uvm_fatal("WB", "Dirty writeback differs from independent model")
          memory[wb_address[d.txn]] = wb_data[d.txn];
          if (d.resp == 1) shared_writebacks++;
        end
        writebacks++;
        wb_address.delete(d.txn);
        wb_data.delete(d.txn);
        wb_seen.delete(d.txn);
      end
    end else if (d.opcode == 1) begin
      if (forbid_early_snoop && fill_packets < partial_goal)
        `uvm_fatal("SNOOP_EARLY", "Same-line snoop responded before the final fill packet")
      if (d.txn != snoop_id || !(&d.be)) `uvm_fatal("SNOOP", "Unexpected snoop data")
      if (snoop_seen[index]) `uvm_fatal("SNOOP", "Duplicate snoop DataID")
      snoop_data[index*256+:256] = d.data;
      snoop_seen[index] = 1;
      if (&snoop_seen) begin
        if (rnf_ref_read(model, snoop_address, golden) || snoop_data !== golden)
          `uvm_fatal("SNOOP", "Snoop differs from golden CPU data")
        memory[snoop_address] = snoop_data;
        snoops++;
      end
    end else `uvm_fatal("DAT", "Unexpected RN-F DAT opcode")
  endfunction
  task monitor();
    forever begin
      @(posedge vif.clock);
      if (!vif.reset) begin
        cycle++;
        if (vif.outstanding > peak) peak = vif.outstanding;
        if (vif.rx_dat_valid && vif.rx_dat_ready) fill_packets++;
        if (vif.req_valid && vif.req_ready) consume_req(req_t'(vif.req));
        if (vif.dat_valid && vif.dat_ready) consume_data(dat_t'(vif.dat));
        if (vif.rsp_valid && vif.rsp_ready) begin
          rsp_t r = rsp_t'(vif.rsp);
          if (r.opcode != 2 && r.opcode != 1) `uvm_fatal("RSP", "Unexpected RN-F RSP opcode")
          if (r.src != 1 || r.tgt != 64) `uvm_fatal("RSP", "RN-F response endpoint mismatch")
          if (r.opcode == 2) begin
            if (!ack_pending.exists(r.txn))
              `uvm_fatal("ACK", "CompAck did not retain the independent Home DBID")
            ack_pending.delete(r.txn);
          end else begin
            if (forbid_early_snoop && fill_packets < partial_goal)
              `uvm_fatal("SNOOP_EARLY", "Same-line snoop responded before the final fill packet")
            if (expect_snoop_data)
              `uvm_fatal("SNOOP_DATA", "Cached RetToSrc snoop did not return data")
            if (r.txn != snoop_id)
              `uvm_fatal("SNOOP", "Dataless snoop response has wrong transaction")
            snoops++;
          end
        end
        if (vif.result_valid && vif.result_ready) begin
          bit [63:0] golden;
          bit error;
          if (!expected.size()) `uvm_fatal("RESULT", "Unexpected completion")
          golden = expected.pop_front();
          error  = expected_errors.pop_front();
          if (vif.result_error != error || vif.result_data !== golden)
            `uvm_fatal("RESULT", $sformatf(
                       "completion %0d got=%h expected=%h", checked, vif.result_data, golden))
          checked++;
        end
        if (rsp_active && vif.rx_rsp_ready) rsp_active = 0;
        if (dat_active && vif.rx_dat_ready) dat_active = 0;
      end
      @(negedge vif.clock);
      vif.req_ready = !vif.reset && !hold_req && cycle % 7 >= 2;
      vif.rsp_ready = !vif.reset && !hold_rsp && cycle % 11 >= 5;
      vif.dat_ready = !vif.reset && !hold_output_dat && cycle % 13 >= 6;
      vif.result_ready = !vif.reset && cycle % 9 >= 3;
      if (!rsp_active && rx_rsp_queue.size()) begin
        vif.rx_rsp = rx_rsp_queue.pop_front();
        rsp_active = 1;
      end
      if (!dat_active && !hold_fill && !hold_dat && rx_dat_queue.size()) begin
        if (reverse_transactions) vif.rx_dat = rx_dat_queue.pop_back();
        else vif.rx_dat = rx_dat_queue.pop_front();
        dat_active = 1;
      end
      vif.rx_rsp_valid = rsp_active && !vif.reset;
      vif.rx_dat_valid = dat_active && !vif.reset;
    end
  endtask
  task issue(longint unsigned address, bit write = 0, bit [63:0] value = 0, int atomic = 0,
             bit [7:0] mask = '1, bit sc_success = 0, bit expect_error = 0, bit atomic_word = 0);
    bit [63:0] answer;
    @(negedge vif.clock);
    vif.access_addr = address;
    vif.access_write = write;
    vif.access_data = value;
    vif.access_atomic = atomic;
    vif.access_atomic_word = atomic_word;
    vif.access_mask = mask;
    vif.access_valid = 1;
    do @(posedge vif.clock); while (!vif.access_ready);
    answer = rnf_ref_access(model, address, write, value, atomic, atomic_word, mask, sc_success,
                            expect_error);
    expected.push_back(answer);
    expected_errors.push_back(expect_error);
    submitted++;
    @(negedge vif.clock);
    vif.access_valid = 0;
  endtask
  task settle();
    wait (checked + cancelled_cpu == submitted);
    while (ack_pending.num() != 0 || wb_address.num() != 0) @(negedge vif.clock);
    repeat (4) @(negedge vif.clock);
  endtask
  task access (longint unsigned address, bit write = 0, bit [63:0] value = 0, int atomic = 0,
               bit [7:0] mask = '1, bit sc_success = 0, bit expect_error = 0, bit atomic_word = 0);
    issue(address, write, value, atomic, mask, sc_success, expect_error, atomic_word);
    settle();
  endtask
  task snoop_line(longint unsigned address, int opcode, bit ret_source = 0);
    snp_t s = '0;
    int   before_count = snoops;
    expect_snoop_data = ret_source;
    snoop_id = snoop_allocations % 2 ? snoop_allocations / 2 : 4095 - snoop_allocations / 2;
    snoop_allocations++;
    s.src = 64;
    s.txn = snoop_id;
    s.addr = address >> 3;
    s.opcode = opcode;
    s.do_not_go_sd = 1;
    s.ret_to_src = ret_source;
    snoop_address = address;
    snoop_seen = 0;
    @(negedge vif.clock);
    vif.snp = s;
    vif.snp_valid = 1;
    do @(posedge vif.clock); while (!vif.snp_ready);
    @(negedge vif.clock);
    vif.snp_valid = 0;
    wait (snoops > before_count);
    repeat (4) @(negedge vif.clock);
  endtask
  task partial_fill_snoop(longint unsigned target, longint unsigned probe, bit reverse);
    int before_packets = fill_packets;
    int before_snoops = snoops;
    reverse_transactions = reverse;
    issue(target);
    wait (fill_packets == before_packets + 1);
    hold_dat = 1;
    partial_goal = before_packets + 2;
    forbid_early_snoop = target == probe;
    fork
      snoop_line(probe, 4, 1);
      begin
        repeat (20) @(negedge vif.clock);
        if (target != probe && snoops == before_snoops)
          `uvm_fatal("SNOOP_INDEPENDENT",
                     "Different-line snoop waited for an unrelated final fill packet")
        hold_dat = 0;
      end
    join
    forbid_early_snoop   = 0;
    reverse_transactions = 0;
    settle();
  endtask
  task reset_pending_reads();
    bit [511:0] line;
    hold_fill = 1;
    issue(1024);
    issue(1088);
    wait (ack_pending.num() == 2);
    if (vif.outstanding != 2 || expected.size() != 2)
      `uvm_fatal("RESET_PENDING", "Reset did not start with two pending reads")
    @(negedge vif.clock);
    vif.reset = 1;
    // Coordinated reset cancels unretired CPU work and both endpoints' CHI work.
    cancelled_cpu += expected.size();
    expected.delete();
    expected_errors.delete();
    ack_pending.delete();
    rx_dat_queue.delete();
    rx_rsp_queue.delete();
    rsp_active = 0;
    dat_active = 0;
    vif.access_addr = 2048;
    vif.access_write = 0;
    vif.access_data = 0;
    vif.access_atomic = 0;
    vif.access_atomic_word = 0;
    vif.access_mask = '1;
    vif.access_valid = 1;
    repeat (4) @(negedge vif.clock);
    if (vif.outstanding != 0) `uvm_fatal("RESET_PENDING", "Reset retained CPU retirement entries")
    // VALID may remain asserted through reset. Transfers count only outside reset;
    // the held command must produce exactly one post-reset completion.
    vif.reset = 0;
    hold_fill = 0;
    do @(posedge vif.clock); while (!vif.access_ready);
    if (rnf_ref_read(model, 2048, line)) `uvm_fatal("MODEL", "Reset probe is uninitialized")
    expected.push_back(line[63:0]);
    expected_errors.push_back(0);
    submitted++;
    @(negedge vif.clock);
    vif.access_valid = 0;
    settle();
  endtask
  task idle_metadata();
    // Invalid payload is unconstrained; changing it must not create a transaction.
    for (int bank = 0; bank < 2; bank++) begin
      for (int field = 0; field < 6; field++) begin
        rsp_t r = '0;
        dat_t d = '0;
        snp_t probe = '0;
        r.src = 64;
        r.tgt = 1;
        r.txn = bank;
        d.opcode = 4;
        d.src = 64;
        d.tgt = 1;
        d.txn = bank;
        d.home = 64;
        probe.src = 64;
        probe.opcode = 4;
        case (field)
          0: begin
            r.tgt = 0;
            d.opcode = 0;
            probe.src = 0;
          end
          1: begin
            r.src = 0;
            d.src = 0;
            probe.pas = 1;
          end
          2: begin
            r.txn = bank ^ 1;
            d.tgt = 0;
            probe.fwd_nid = 1;
          end
          3: begin
            r.error = 3;
            d.home = 0;
            probe.fwd_txn = 1;
          end
          4: begin
            r.txn = 4095;
            d.txn = 4095;
            probe.opcode = 0;
          end
          5: begin
            d.data_id = 3;
            d.dbid = 16'hf000;
          end
        endcase
        @(negedge vif.clock);
        vif.rx_rsp = r;
        vif.rx_dat = d;
        vif.snp = probe;
        vif.access_addr = bank * 64 + 1;
        vif.access_atomic = 15;
        vif.access_atomic_word = 0;
        vif.access_write = 1;
        vif.access_mask = 0;
        repeat (2) @(negedge vif.clock);
      end
    end
    for (int bit_index = 0; bit_index < $bits(dat_t); bit_index++) begin
      @(negedge vif.clock);
      vif.rx_rsp = rsp_t'(rsp_t'(1) << bit_index);
      vif.rx_dat = dat_t'(dat_t'(1) << bit_index);
      vif.snp = snp_t'(snp_t'(1) << bit_index);
      vif.access_addr = 44'(1) << bit_index;
      vif.access_data = 64'(1) << bit_index;
      vif.access_mask = 8'(1) << bit_index;
      vif.access_atomic = 4'(1) << bit_index;
      vif.access_atomic_word = bit_index % 2;
      vif.access_write = bit_index % 2;
      repeat (2) @(negedge vif.clock);
      if (vif.req_valid || vif.rsp_valid || vif.dat_valid || vif.result_valid || vif.outstanding)
        `uvm_fatal("IDLE_PAYLOAD", "Invalid walking-bit payload created CPU or CHI work")
    end
    if (vif.req_valid || vif.rsp_valid || vif.dat_valid || vif.result_valid || vif.outstanding)
      `uvm_fatal("IDLE_PAYLOAD", "Invalid payload created CPU or CHI work")
    vif.rx_rsp = '0;
    vif.rx_dat = '0;
    vif.snp = '0;
    vif.access_addr = 0;
    vif.access_atomic = 0;
    vif.access_atomic_word = 0;
    vif.access_write = 0;
    vif.access_mask = '1;
  endtask
  task execute();
    vif.reset = 1;
    vif.access_valid = 0;
    vif.access_atomic_word = 0;
    vif.snp_valid = 0;
    vif.rx_rsp_valid = 0;
    vif.rx_dat_valid = 0;
    vif.req_ready = 0;
    vif.rsp_ready = 0;
    vif.dat_ready = 0;
    vif.result_ready = 0;
    for (int i = 0; i < 256; i++) begin
      bit [511:0] initial_line;
      for (int j = 0; j < 8; j++)
      initial_line[j*64 +: 64] = (i%2 ? 64'h0f0f0f0f0f0f0f0f : 64'hf0f0f0f0f0f0f0f0) ^ 64'(i*8+j);
      memory[i*64] = initial_line;
      if (rnf_ref_write(model, i * 64, initial_line, '1))
        `uvm_fatal("MODEL", "Initialization failed")
    end
    repeat (4) @(negedge vif.clock);
    vif.reset = 0;
    idle_metadata();
    fork
      monitor();
    join_none
    hold_fill = 1;
    issue(0);
    issue(64);
    repeat (10) @(negedge vif.clock);
    if (vif.outstanding != 2) `uvm_fatal("PARALLEL", "Both cache banks did not remain in flight")
    reverse_transactions = 1;
    hold_fill = 0;
    settle();
    reverse_transactions = 0;
    for (int line_index = 2; line_index < 8; line_index++) begin
      access (line_index * 64);
    end
    reset_pending_reads();
    for (int word = 1; word < 8; word++) begin
      access (word * 8);
      access (word * 8, 1, 64'h55aa55aa55aa55aa ^ 64'(word), 0, 8'hff >> word);
    end
    access (0, 1, 'h1122334455667788, 0, 'h0f);
    snoop_line(0, 4);  // downgrade dirty to shared
    access (0);
    access (0, 1, 'h8877665544332211);
    access (512);  // conflicting dirty victim
    access (0);
    for (int bank = 0; bank < 2; bank++) begin
      for (int op = 1; op <= 9; op++) begin
        // Exercise the independent hit and fill AMO datapaths with both comparisons.
        access (bank * 64, 1, 0);
        access (bank * 64, 0, 3, op);
        access (bank * 64, 0, 64'hfffffffffffffffe, op);
        snoop_line(bank * 64, 4);
        access (bank * 64, 0, 3, op);
        snoop_line(bank * 64, 4);
        access (bank * 64, 0, 64'hfffffffffffffffe, op);
        if (op >= 6) begin
          for (int refill = 0; refill < 2; refill++) begin
            access (bank * 64, 1, '1);
            if (refill) snoop_line(bank * 64, 4);
            access (bank * 64, 0, 0, op);
            access (bank * 64, 1, 0);
            if (refill) snoop_line(bank * 64, 4);
            access (bank * 64, 0, '1, op);
          end
        end
      end
      access (bank * 64, 0, 0, 10);
      access (bank * 64, 0, 99, 11, '1, 1);  // successful SC hit
      snoop_line(bank * 64, 4);
      access (bank * 64, 0, 0, 10);
      access (bank * 64, 0, 101, 11, '1, 1);  // successful SC fill
      access (bank * 64, 0, 0, 12);
    end
    // AMO.W uses an independent u32/i32 reference and must preserve the other lane.
    for (int bank = 0; bank < 2; bank++) begin
      for (int lane = 0; lane < 2; lane++) begin
        longint unsigned address = bank * 64 + lane * 4;
        bit [31:0] lhs_values[4] = '{32'hffffffff, 32'h80000000, 32'h7fffffff, 32'h0};
        bit [31:0] rhs_values[4] = '{32'h1, 32'h7fffffff, 32'h80000000, 32'hffffffff};
        for (int op = 1; op <= 9; op++) begin
          for (int boundary = 0; boundary < 4; boundary++) begin
            bit [63:0] initial_value = 64'h5aa55aa5a55aa55a;
            initial_value[lane*32+:32] = lhs_values[boundary];
            access (bank * 64, 1, initial_value);
            if (boundary % 2) snoop_line(bank * 64, 4);  // exercise refill permission upgrade
            // High operand bits are intentionally unrelated to the 32-bit operation.
            access (address, 0, {32'hdeadbeef, rhs_values[boundary]}, op, '1, 0, 0, 1);
            access (bank * 64);  // independently check both lane update and neighbor preservation
          end
        end
        access (address, 0, 0, 10, '1, 0, 0, 1);
        access (address, 0, 64'habcdef0180000001, 11, '1, 1, 0, 1);
        access (bank * 64);
        snoop_line(bank * 64, 4);
        access (address, 0, 0, 10, '1, 0, 0, 1);
        access (address, 0, 64'hffffffffffffffff, 11, '1, 1, 0, 1);
        access (bank * 64);
        access (address, 0, 0, 10, '1, 0, 0, 1);
        snoop_line(bank * 64, 7);
        access (address, 0, 99, 11, '1, 0, 0, 1);
        access (address, 0, 0, 10, '1, 0, 0, 1);
        access (bank * 64 + (1 - lane) * 4, 0, 77, 11, '1, 0, 0, 1);
        access (address, 0, 88, 11, '1, 0, 0, 1);  // failed other-lane SC consumes LR
        access (address, 0, 0, 10, '1, 0, 0, 1);
        access ((1 - bank) * 64, 0, 77, 11, '1, 0, 0, 1);
        access (address, 0, 88, 11, '1, 0, 0, 1);  // failed other-bank SC consumes LR
        access (address, 0, 0, 10, '1, 0, 0, 1);
        access ((1 - bank) * 64, 0, 0, 10);  // later LR.D replaces LR.W across banks
        access (address, 0, 88, 11, '1, 0, 0, 1);
      end
      access (bank * 64, 0, 0, 10, '1, 0, 0, 1);
      access (bank * 64, 0, 88, 11);  // same address, mismatched reservation width
      access (bank * 64, 0, 0, 10);
      access (bank * 64, 0, 88, 11, '1, 0, 0, 1);
      access (bank * 64, 0, 0, 10, '1, 0, 0, 1);
      snoop_line((1 - bank) * 64, 7);  // unrelated bank invalidation preserves reservation
      access (bank * 64, 0, 99, 11, '1, 1, 0, 1);
      access (bank * 64);
    end
    // Every word position uses both lane selectors in the hit and refill merge datapaths.
    for (int bank = 0; bank < 2; bank++) begin
      for (int word_index = 0; word_index < 8; word_index++) begin
        longint unsigned word_address = bank * 64 + word_index * 8;
        access (word_address, 1, 64'h800000007fffffff);
        for (int lane = 0; lane < 2; lane++) begin
          access (word_address + lane * 4, 0, 64'hdeadbeef00000001, 2, '1, 0, 0, 1);
          access (word_address);
          snoop_line(bank * 64, 4);
          access (word_address + lane * 4, 0, 64'hdeadbeefffffffff, 2, '1, 0, 0, 1);
          access (word_address);
        end
      end
    end
    access (576);  // dirty eviction through the second bank
    access (64);
    access (64, 0, 0, 10);
    access (64, 0, 99, 11, '1, 1);
    access (64, 0, 0, 10);
    snoop_line(64, 7);  // invalidate consumes LR
    access (64, 0, 123, 11);
    access (0, 0, 0, 10);
    access (64, 0, 123, 11);  // SC to other bank fails and consumes reservation
    access (0, 0, 123, 11);
    access (0, 0, 0, 12);
    snoop_line(448, 4);  // clean cached Shared line: dataless response
    snoop_line(1536, 9);  // absent line: dataless Invalid response
    fail_read = 2;
    access (768, 0, 0, 0, '1, 0, 1);
    fail_read = 0;
    access (768);  // errored fill must not leave a valid cache line
    access (512);  // write back the bank0 AMO/SC result before reset invalidates L1
    // Coordinated, quiescent reset invalidates L1; backing contents persist.
    @(negedge vif.clock);
    vif.reset = 1;
    repeat (4) @(negedge vif.clock);
    vif.reset = 0;
    issue(0);
    access (64);
    // Every legal high address bit must reach the tag compare and eviction path.
    for (int bit_index = 9; bit_index < 44; bit_index++) begin
      for (int cache_index = 0; cache_index < 8; cache_index++) begin
        longint unsigned high = (64'(1) << bit_index) | (64'(cache_index) << 6);
        bit [511:0] high_line = {8{(bit_index % 2 ? 64'h0123456789abcdef : ~64'h0123456789abcdef) ^ high}};
        memory[high] = high_line;
        if (rnf_ref_write(model, high, high_line, '1)) `uvm_fatal("MODEL", "High-tag init failed")
        access (high, 0, 0, 10);  // full-address LR reservation in each bank
        snoop_line(high, 4, 1);  // all entry indices and high Snoop addresses, full clean data
        access (high + 56, 1,
                (bit_index % 2 ? 64'h0123456789abcdef : ~64'h0123456789abcdef) ^ high);
        access (high + 8, 1,
                (bit_index % 2 ? ~64'h0123456789abcdef : 64'h0123456789abcdef) ^ high);  // hit-write R3 read port
        access (high);  // R0 hit-read port must observe the complementary full line
        access (cache_index * 64);  // dirty high-tag copyback retains the full address
      end
    end
    for (int bank = 0; bank < 2; bank++) begin
      access (bank * 64, 1, 64'h1234567890abcdef);
      hold_req = 1;
      issue(512 + bank * 64);
      begin
        req_t pending_request;
        do begin
          @(negedge vif.clock);
          pending_request = req_t'(vif.req);
        end while (!vif.req_valid || pending_request.opcode != 'h1b);
      end
      snoop_line(bank * 64, 7);  // revoke dirty victim while its WriteBackFull is queued
      begin
        bit [511:0] replacement = {8{64'hfedcba9876543210}};
        memory[bank*64] = replacement;
        if (rnf_ref_write(model, bank * 64, replacement, '1))
          `uvm_fatal("MODEL", "Remote write failed")
      end
      hold_req = 0;
      settle();
      access (bank * 64);  // cancelled copyback must not overwrite the newer line
    end
    if (cancelled_writebacks == 0)
      `uvm_fatal("CANCEL", "Queued writeback cancellation did not execute")
    fail_read = 2;
    access (832, 0, 0, 0, '1, 0, 1);  // second bank error path
    fail_read = 0;
    access (832);
    access (448, 0, 0, 10);
    snoop_line(448, 4, 1);  // clean RetToSrc data must preserve LR reservation
    access (448, 0, 44, 11, '1, 1);
    for (int bank = 0; bank < 2; bank++) begin
      access (bank * 64, 1, 64'h0123456789abcdef);
      hold_req = 1;
      issue(512 + bank * 64);
      begin
        req_t pending_request;
        do begin
          @(negedge vif.clock);
          pending_request = req_t'(vif.req);
        end while (!vif.req_valid || pending_request.opcode != 'h1b);
      end
      snoop_line(bank * 64, 4);  // queued dirty victim is downgraded to Shared before copyback
      hold_req = 0;
      settle();
      access (bank * 64);
      for (int entry = 0; entry < 4; entry++) begin
        longint unsigned address = entry * 128 + bank * 64;
        access (address);
        snoop_line(address, 4);
        access (address, 0, 0, 10);
        hold_req = 1;
        issue(address, 0, 64'habcdef,
              11);  // initially valid LR, SC must fail after the intervening snoop
        begin
          req_t pending_request;
          do begin
            @(negedge vif.clock);
            pending_request = req_t'(vif.req);
          end while (!vif.req_valid || pending_request.opcode != 'h07);
        end
        snoop_line(address, 7);
        hold_req = 0;
        settle();
        access (address);  // failed SC fill acquired UC but must not modify the word
        access (address, 1, 64'h123456789abcdef0);  // clean UC hit becomes dirty
      end
    end
    for (int order = 0; order < 2; order++) begin
      access (0, 1, 64'h55aa55aa55aa55aa);
      access (64, 1, 64'haa55aa55aa55aa55);
      hold_output_dat = 1;
      issue(3072 + order * 64);
      issue(3136 - order * 64);  // both bank arrival orders
      repeat (60) @(negedge vif.clock);
      hold_output_dat = 0;
      settle();
    end
    // Fill both output request and response queues under sustained legal backpressure.
    hold_req = 1;
    hold_rsp = 1;
    issue(2560);
    issue(2624);
    repeat (30) @(negedge vif.clock);
    hold_req = 0;
    repeat (60) @(negedge vif.clock);
    fork
      snoop_line(16000, 9);  // third response meets two queued CompAcks
      begin
        repeat (20) @(negedge vif.clock);
        hold_rsp = 0;
      end
    join
    settle();
    for (int bank = 0; bank < 2; bank++) begin
      for (int entry = 0; entry < 4; entry++) begin
        longint unsigned address = entry * 128 + bank * 64;
        snoop_line(address, 7);
        snoop_line(address, 4);  // absent Shared snoop at each directory entry
        fail_read = entry % 2 ? 3 : 2;
        mixed_data_error = entry == 2;
        access (address, 0, 0, entry % 2 ? 0 : 10, '1, 0, 1);
        fail_read = 0;
        mixed_data_error = 0;
        access (((entry + 1) % 4) * 128 + bank * 64);  // a hit clears the previous response error
        access (address);
        access (address, 0, 0, 10);
        access (address);  // ordinary hit must retain LR
        access (address, 0, 11, 11, '1, 1);
      end
    end
    for (int bank = 0; bank < 2; bank++) begin
      access (bank * 64, 0, 0, 10);
      snoop_line(128 + bank * 64, 7);  // unrelated-line invalidation must preserve LR
      access (bank * 64, 0, 77, 11, '1, 1);
      access (bank * 64, 0, 0, 0, 0);  // a normal read does not consume byte enables
      fork
        issue(bank * 64);
        begin
          repeat (2) @(posedge vif.clock);
          snoop_line(256 + bank * 64, 4);  // snoop overlaps lookup and temporarily stops it
        end
      join
      settle();
    end
    access (128);  // independent-line probe must be resident after invalidation scenarios
    partial_fill_snoop(1024, 1024, 0);
    partial_fill_snoop(1536, 1536, 1);  // opposite DataID arrival order
    partial_fill_snoop(2048, 128, 0);  // same bank, independent cache line
    for (int bank = 0; bank < 2; bank++) begin
      access (bank * 64 + 56, 1, '1);
      access (bank * 64, 1, '1);  // full-width hit merge input includes the high word
      access (bank * 64 + 56, 1, 0);
      access (bank * 64, 1, 0);
      for (int word = 0; word < 8; word++) begin
        access (bank * 64 + word * 8, 0, 0, 10);
        access (bank * 64 + word * 8, 1, '1, 0, 8'hfe);
        access (bank * 64 + word * 8, 1, 0, 0, 8'h00);
        access (bank * 64 + word * 8);
      end
      access (bank * 64, 0, 0, 10);  // release high word-offset reservation bits
    end
    for (int bank = 0; bank < 2; bank++) begin
      for (int miss = 0; miss < 2; miss++) begin
        access (bank * 64);
        for (int bit_index = 0; bit_index <= 32; bit_index++) begin
          int unsigned seed = bit_index == 32 ? '1 : (32'(1) << bit_index) - 1;
          int unsigned other, expected_counter = seed + 1;
          longint unsigned address = miss ? (bit_index % 2 ? 1024 : 512) + bank * 64 : bank * 64;
          @(negedge vif.clock);
          other = vif.counter_value(!bank, miss);
          vif.seed_counter(bank, miss, seed);
          access (address);
          if (vif.counter_value(bank, miss) !== expected_counter)
            `uvm_fatal("COUNTER", "Real hit/miss did not increment the seeded statistical boundary")
          if ((miss ? vif.misses : vif.hits) !== 32'(other + expected_counter))
            `uvm_fatal("COUNTER_SUM", "Bank sum did not retain modulo-32bit statistics")
        end
      end
    end
    begin
      longint unsigned a = 24576;
      bit [511:0] fresh = '0;
      int before_alloc;
      memory[a] = fresh;
      if (rnf_ref_write(model, a, fresh, '1))
        `uvm_fatal("MODEL", "CMO fixture initialization failed")
      access (a, 1, 64'h1122334455667788);
      snoop_line(a, 8);  // SnpCleanShared: pass dirty data, retain a clean shared copy.
      before_alloc = home_allocations;
      access (a);
      if (home_allocations != before_alloc)
        `uvm_fatal("CLEAN_SHARED", "SnpCleanShared invalidated a resident clean copy")
      access (a, 1, 64'hdeadbeef01234567);
      fresh[63:0] = 64'hcafebabe87654321;
      memory[a]   = fresh;
      if (rnf_ref_write(model, a, fresh, '1))
        `uvm_fatal("GOLDEN", "DMA test write was outside physical memory")
      snoop_line(a, 10);  // SnpMakeInvalid must not return stale dirty data.
      if (memory[a] !== fresh)
        `uvm_fatal("MAKE_INVALID", "Stale dirty data overwrote newer DMA memory")
      before_alloc = home_allocations;
      access (a);
      if (home_allocations != before_alloc + 1)
        `uvm_fatal("INVALID_REFILL", "SnpMakeInvalid did not invalidate the resident line")
      snoop_line(a, 8);  // Clean shared copy: dataless retained response.
      snoop_line(a, 10);
      snoop_line(a, 10);  // Absent line: dataless invalid response.
      `uvm_info("CMO", "SnpCleanShared retention and SnpMakeInvalid dirty discard checked", UVM_LOW)
    end
    if (shared_writebacks == 0)
      `uvm_fatal("SHARED_WB", "Downgraded victim did not return Shared copyback")
    if (peak < 2 || writebacks == 0 || snoops < 2 || expected.size())
      `uvm_fatal("COVERAGE", "Required parallel, dirty eviction or snoop case not exercised")
    `uvm_info(
        "RNF",
        $sformatf(
            "Checked %0d CPU completions, peak=%0d writebacks=%0d snoops=%0d cancelled=%0d shared=%0d reset_cancelled=%0d",
            checked, peak, writebacks, snoops, cancelled_writebacks, shared_writebacks,
            cancelled_cpu), UVM_LOW)
    rnf_ref_destroy(model);
  endtask
endclass
