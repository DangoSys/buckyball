class scenario_sequence extends uvm_sequence;
  `uvm_object_utils(scenario_sequence)
  localparam int STALL_READS = 15 / `CHI_BEATS + 8;
  env test_env;
  protocol_model state;
  virtual chi_sram_if vif;
  string scenario;
  function new(string name = "scenario_sequence");
    super.new(name);
  endfunction
  task packet(int port, logic [`CHI_DAT_BITS-1:0] payload, bit sc_success = 0, bit failed = 0);
    packet_sequence seq = packet_sequence::type_id::create("packet");
    seq.item = packet_item::type_id::create("item");
    seq.item.payload = payload;
    seq.item.sc_success = sc_success;
    seq.item.failed = failed;
    seq.start(test_env.bus.seqr[port]);
  endtask
  task reset_link();
    state.idle.rx_active_req = 0;
    state.idle.tx_active_ack = 0;
    state.idle.req_pend = 0;
    state.idle.dat_pend = 0;
    test_env.reset.pulse(4);
    repeat (3) @(negedge vif.clock);
    state.idle.rx_active_req = 1;
    wait (state.sampled.tx_active_req);
    repeat (3) @(negedge vif.clock);
    state.idle.tx_active_ack = 1;
    state.idle.req_pend = 1;
    state.idle.dat_pend = 1;
    repeat (4) @(negedge vif.clock);
    evidence::check(state.sampled.rx_active_ack && state.sampled.tx_active_req,
                    "CHI SRAM activation failed");
  endtask

  task send_req(req_t value);
    packet(0, value);
  endtask

  task send_dat(dat_t value);
    packet(1, value);
  endtask

  function req_t request(int opcode, int txn, longint unsigned addr, int source = 3);
    req_t r = '0;
    r.opcode = opcode;
    r.txn = txn;
    r.addr = addr;
    r.size = 6;
    r.src = (txn ^ source) & 127;
    r.tgt = 1;
    r.return_nid = (txn ^ 85) & 127;
    r.return_txn = txn + 100;
    r.allow_retry = txn % 2;
    r.mem_attr = (txn % 3 == 0 ? 0 : txn % 3 == 1 ? 4 : 12) | (txn % 2);
    r.likely_shared = txn % 2;
    r.lpid = txn % 32;
    r.qos = txn % 16;
    r.trace_tag = txn % 2;
    return r;
  endfunction
  task response(req_t r, int opcode, output rsp_t out_rsp);
    bit found = 0;
    while (!found) begin
      @(negedge vif.clock);
      foreach (state.responses[i]) begin
        if (state.responses[i].txn == r.txn && state.responses[i].tgt == r.src) begin
          out_rsp = state.responses[i];
          state.responses.delete(i);
          found = 1;
          break;
        end
      end
    end
    if (out_rsp.opcode != opcode || out_rsp.src != 1 || out_rsp.qos != r.qos ||
        out_rsp.trace_tag != r.trace_tag)
      `uvm_fatal("RESPONSE", $sformatf("Bad response txn=%0d opcode=%0h", r.txn, out_rsp.opcode))
    state.checks++;
    evidence::checks++;
  endtask
  function dat_t write_beat(req_t r, rsp_t dbid_rsp, int beat, bit [`CHI_DATA_BITS-1:0] data,
                            bit [`CHI_BE_BITS-1:0] mask = '1);
    dat_t d = '0;
    d.tgt = 1;
    d.src = r.src;
    d.txn = dbid_rsp.dbid;
    d.qos = r.qos;
    d.dbid = {4'b0, r.txn};
    d.line_id = r.addr[11:6];
    d.trace_tag = r.trace_tag;
    d.opcode = 3;
    d.data_id = beat * `CHI_DATA_ID_STEP;
    d.data = data;
    d.be = mask;
    return d;
  endfunction
  task write_data(req_t r, rsp_t dbid_rsp, bit [511:0] data, bit [63:0] mask);
    for (int beat = `CHI_BEATS - 1; beat >= 0; beat--)
      send_dat(write_beat(
               r,
               dbid_rsp,
               beat,
               data[beat*`CHI_DATA_BITS+:`CHI_DATA_BITS],
               mask[beat*`CHI_BE_BITS+:`CHI_BE_BITS]
               ));
  endtask
  task write_line(int txn, longint unsigned addr, bit [511:0] data, bit [63:0] mask = '1);
    req_t r = request(mask == '1 ? 'h1d : 'h1c, txn, addr);
    rsp_t s;
    byte unsigned expected_error;
    send_req(r);
    response(r, 6, s);
    write_data(r, s, data, mask);
    response(r, 4, s);
    expected_error = chi_ref_write(state.reference, addr, data, mask);
    if (s.error != (expected_error ? 3 : 0)) `uvm_fatal("ERROR", "Wrong write error")
    state.checks++;
    evidence::checks++;
  endtask
  task read_line(int txn, longint unsigned addr);
    req_t r = request(4, txn, addr);
    dat_t d;
    bit [511:0] golden;
    bit [`CHI_BEATS-1:0] seen = 0;
    byte unsigned error = chi_ref_read(state.reference, addr, golden);
    send_req(r);
    while (!(&seen)) begin
      bit found = 0;
      @(negedge vif.clock);
      foreach (state.data_responses[i]) begin
        if (state.data_responses[i].txn == r.return_txn) begin
          d = state.data_responses[i];
          state.data_responses.delete(i);
          found = 1;
          break;
        end
      end
      if (found) begin
        int b = d.data_id / `CHI_DATA_ID_STEP;
        if ((d.data_id % `CHI_DATA_ID_STEP != 0 || b >= `CHI_BEATS) || seen[b] || d.opcode != 4 ||
            d.tgt != r.return_nid || d.src != 1 || d.home != r.src || d.dbid != r.txn ||
            d.error != (error ? 3 : 0) || d.qos != r.qos || d.trace_tag != r.trace_tag ||
            d.data !== golden[b*`CHI_DATA_BITS +: `CHI_DATA_BITS])
          `uvm_fatal("READ", $sformatf("Bad read data txn=%0d DataID=%0d", txn, d.data_id))
        seen[b] = 1;
        state.checks++;
        evidence::checks++;
      end
    end
  endtask
  task scenarios();
    req_t r[8];
    rsp_t dbid[8], s;
    bit [511:0] words[8];
    reset_link();
    if (scenario == "bad_opcode") begin
      send_req(request('h07, 0, 0));
      repeat (20) @(negedge vif.clock);
      `uvm_fatal("NEGATIVE", "Unsupported opcode did not terminate RTL")
    end else if (scenario == "duplicate_data") begin
      r[0] = request('h1d, 0, 0);
      send_req(r[0]);
      response(r[0], 6, s);
      begin
        dat_t d = '0;
        d.tgt = 1;
        d.src = r[0].src;
        d.txn = s.dbid;
        d.opcode = 3;
        d.data_id = 0;
        d.be = '1;
        send_dat(d);
        send_dat(d);
      end
      repeat (20) @(negedge vif.clock);
      `uvm_fatal("NEGATIVE", "Duplicate DataID did not terminate RTL")
    end else begin
      // Fill all eight transaction slots, then interleave reverse-order DataIDs.
      for (int i = 0; i < 8; i++) begin
        for (int b = 0; b < 64; b++) words[i][b*8+:8] = (i * 37 + b);
        r[i] = request('h1d, i, i * 64, i % 2 ? 5 : 3);
        send_req(r[i]);
        response(r[i], 6, dbid[i]);
      end
      if (state.sampled.outstanding != 8) `uvm_fatal("MSHR", "Failed to occupy all slots")
      for (int beat = `CHI_BEATS - 1; beat >= 0; beat--) begin
        for (int i = 7; i >= 0; i--) begin
          send_dat(write_beat(r[i], dbid[i], beat, words[i][beat*`CHI_DATA_BITS+:`CHI_DATA_BITS], '1
                   ));
        end
      end
      for (int i = 0; i < 8; i++) begin
        response(r[i], 4, s);
        if (s.error != 0 || chi_ref_write(state.reference, i * 64, words[i], '1) != 0)
          `uvm_fatal("WRITE", "Initial full write failed")
      end
      for (int i = 0; i < 8; i++) read_line(32 + i, i * 64);
      // Every live slot must merge byte enables independently, including interleaved beats.
      for (int phase_index = 0; phase_index < 2; phase_index++) begin
        bit [63:0] mask = phase_index ? 64'haaaaaaaaaaaaaaaa : 64'h5555555555555555;
        for (int i = 0; i < 8; i++) begin
          r[i] = request('h1c, 140 + phase_index * 8 + i, i * 64, i % 2 ? 5 : 3);
          send_req(r[i]);
          response(r[i], 6, dbid[i]);
        end
        for (int beat = `CHI_BEATS - 1; beat >= 0; beat--) begin
          for (int i = 7; i >= 0; i--) begin
            send_dat(write_beat(
                     r[i],
                     dbid[i],
                     beat,
                     phase_index ? words[i][beat*`CHI_DATA_BITS +: `CHI_DATA_BITS] : ~words[i][beat*`CHI_DATA_BITS +: `CHI_DATA_BITS],
                     mask[beat*`CHI_BE_BITS+:`CHI_BE_BITS]
                     ));
          end
        end
        for (int i = 0; i < 8; i++) begin
          bit [511:0] data = phase_index ? words[i] : ~words[i];
          response(r[i], 4, s);
          if (s.error || chi_ref_write(state.reference, i * 64, data, mask))
            `uvm_fatal("MASK", "Concurrent partial write failed")
        end
        for (int i = 0; i < 8; i++) read_line(240 + phase_index * 8 + i, i * 64);
      end
      `uvm_info("PHASE", "Full-width slot vectors", UVM_LOW)
      // Exercise each real transaction slot's address, opaque IDs and error status.
      // Writes are concurrent, addresses distinct; then verify valid writes by readback.
      for (int polarity = 0; polarity < 2; polarity++) begin
        for (int address_bit = 9; address_bit < 44; address_bit++) begin
          for (int i = 0; i < 8; i++) begin
            longint unsigned high = (64'h1 << address_bit) ^ (polarity ? 64'hffffffffe00 : 0);
            int tag = (1 << ((address_bit + i) % 12)) ^ (polarity ? 'hfff : 0);
            int source = ((1 << ((address_bit + i) % 7)) | 2) ^ (polarity ? 'h7f : 0);
            r[i] = request('h1d, tag, high | ((i ^ (1 << ((address_bit + polarity) % 3))) << 6));
            r[i].src = source;
            r[i].return_nid = source ^ 'h7f;
            r[i].return_txn = tag ^ 'hfff;
            r[i].qos = (address_bit + i) % 16;
            r[i].trace_tag = polarity;
            send_req(r[i]);
            response(r[i], 6, dbid[i]);
          end
          for (int beat = `CHI_BEATS - 1; beat >= 0; beat--) begin
            for (int i = 7; i >= 0; i--) begin
              send_dat(write_beat(
                       r[i],
                       dbid[i],
                       beat,
                       polarity ? words[i][beat*`CHI_DATA_BITS +: `CHI_DATA_BITS] : ~words[i][beat*`CHI_DATA_BITS +: `CHI_DATA_BITS],
                       '1
                       ));
            end
          end
          for (int i = 0; i < 8; i++) begin
            bit [511:0] data = polarity ? words[i] : ~words[i];
            byte unsigned error;
            response(r[i], 4, s);
            error = chi_ref_write(state.reference, r[i].addr, data, '1);
            if (s.error != (error ? 3 : 0))
              `uvm_fatal("ERROR_SLOT", "Wrong error or opaque transaction attribution")
            state.checks++;
            evidence::checks++;
          end
          if (!polarity && address_bit < 14)
            for (int i = 0; i < 8; i++) read_line(600 + address_bit * 8 + i, r[i].addr);
        end
      end
      write_line(50, 0, '1, 64'h5555555555555555);
      write_line(51, 0, '0, 64'haaaaaaaaaaaaaaaa);
      write_line(52, 0, '1, '0);
      read_line(53, 0);
      write_line(54, 255 * 64, '1);
      read_line(55, 255 * 64);
      write_line(56, 256 * 64, '1);
      read_line(57, 256 * 64);
      // Full-width addresses and TxnIDs must not alias into SRAM or DBID slots.
      write_line(4095, 0, '1);
      read_line(3950, 0);
      write_line(58, 64'h80000000000, '1);
      read_line(59, 64'h80000000000);
      `uvm_info("PHASE", "Data credit stall", UVM_LOW)
      // Block output credit return; requests remain live, then resume safely.
      state.grant_enabled = 0;
      for (int i = 0; i < STALL_READS; i++) send_req(request(4, 80 + i, (i % 8) * 64));
      repeat (25) @(negedge vif.clock);
      if (state.sampled.outstanding == 0)
        `uvm_fatal("STALL", "No live request under data credit stall")
      state.grant_enabled = 1;
      // These reads were issued already: compare every returned beat before reset.
      for (int i = 0; i < STALL_READS; i++) begin
        bit [511:0] golden;
        bit [`CHI_BEATS-1:0] seen = 0;
        if (chi_ref_read(state.reference, (i % 8) * 64, golden))
          `uvm_fatal("MODEL", "Unexpected error")
        while (!(&seen)) begin
          @(negedge vif.clock);
          foreach (state.data_responses[j]) begin
            if (state.data_responses[j].txn == 180 + i) begin
              dat_t d = state.data_responses[j];
              int   b = d.data_id / `CHI_DATA_ID_STEP;
              if (seen[b] || d.data !== golden[b*`CHI_DATA_BITS+:`CHI_DATA_BITS])
                `uvm_fatal("STALL", "Stalled read corrupted")
              seen[b] = 1;
              state.checks++;
              evidence::checks++;
              state.data_responses.delete(j);
              break;
            end
          end
        end
      end
      `uvm_info("PHASE", "Read-to-write slot reuse", UVM_LOW)
      // Reallocate every slot to a write after the all-slot read drain.
      for (int i = 0; i < 8; i++) begin
        r[i] = request('h1d, 2800 + i, i * 64);
        send_req(r[i]);
        response(r[i], 6, dbid[i]);
      end
      for (int i = 0; i < 8; i++) write_data(r[i], dbid[i], words[i], '1);
      for (int i = 0; i < 8; i++) begin
        response(r[i], 4, s);
        if (s.error || chi_ref_write(state.reference, i * 64, words[i], '1))
          `uvm_fatal("SLOT_REUSE", "Read-to-write slot reuse failed")
      end
      for (int i = 0; i < 8; i++) read_line(2900 + i, i * 64);
      `uvm_info("PHASE", "Response and request queue saturation", UVM_LOW)
      // Exhaust all15 RSP credits, then fill all8 node slots and all4 RX entries.
      // Distinct TxnIDs from one source and equal TxnIDs from distinct sources are legal.
      begin
        req_t blocked[11];
        int data_order[8] = '{0, 2, 1, 3, 4, 6, 5, 7};
        while (state.rsp_granted != 15) @(negedge vif.clock);
        state.grant_enabled = 0;
        for (int i = 0; i < 8; i++) begin
          r[i] = request('h1d, 3200 + i, i * 64);
          r[i].src = 3;
          send_req(r[i]);
          response(r[i], 6, dbid[i]);
        end
        for (int i = 0; i < 8; i++) begin
          int slot = data_order[i];
          write_data(r[slot], dbid[slot], words[slot], '1);
        end
        // Drain seven initial Comps before new DBID state.responses compete for credits.
        // One-beat writes can otherwise finish their data input before SRAM replies.
        while (state.rsp_granted != 0 || state.responses.size() != 7) @(negedge vif.clock);
        for (int i = 0; i < 11; i++) begin
          blocked[i] = request('h1d, 3400, i * 64);
          blocked[i].src = 10 + i;
          send_req(blocked[i]);
        end
        repeat (30) @(negedge vif.clock);
        if (state.sampled.outstanding != 8 || state.req_credits != 0)
          `uvm_fatal("RSP_STALL", "RSP credit exhaustion did not saturate node and RX queue")
        state.grant_enabled = 1;
        for (int i = 0; i < 8; i++) begin
          response(r[i], 4, s);
          if (s.error || chi_ref_write(state.reference, i * 64, words[i], '1))
            `uvm_fatal("RSP_STALL", "Initial write failed")
        end
        for (int i = 0; i < 11; i++) begin
          response(blocked[i], 6, s);
          write_data(blocked[i], s, words[i%8], '1);
          response(blocked[i], 4, s);
          if (s.error || chi_ref_write(state.reference, i * 64, words[i%8], '1))
            `uvm_fatal("RSP_STALL", "Queued same-TxnID write attributed to wrong source")
        end
        for (int i = 0; i < 11; i++) read_line(3600 + i, i * 64);
      end
      `uvm_info("PHASE", "Isolated upper read slots", UVM_LOW)
      // Hold lower slots in write-data collection to isolate each upper read slot.
      // A second read in that slot exercises arbitration after its own last grant.
      for (int target_slot = 4; target_slot < 8; target_slot++) begin
        for (int i = 0; i < target_slot; i++) begin
          r[i] = request('h1d, 3700 + i, i * 64);
          r[i].src = 3;
          send_req(r[i]);
          response(r[i], 6, dbid[i]);
        end
        read_line(3800 + target_slot * 2, 0);
        read_line(3801 + target_slot * 2, 0);
        for (int i = 0; i < target_slot; i++) begin
          write_data(r[i], dbid[i], words[i], '1);
          response(r[i], 4, s);
          if (s.error || chi_ref_write(state.reference, i * 64, words[i], '1))
            `uvm_fatal("ARBITRATION", "Isolated slot damaged held writes")
        end
      end
      `uvm_info("PHASE", "Source and transaction identity", UVM_LOW)
      // Slot7 stays live while a lower free slot accepts another transaction from its source.
      for (int i = 0; i < 8; i++) begin
        r[i] = request('h1d, 3900 + i, i * 64);
        r[i].src = 3;
        send_req(r[i]);
        response(r[i], 6, dbid[i]);
      end
      for (int i = 0; i < 2; i++) begin
        write_data(r[i], dbid[i], words[i], '1);
        response(r[i], 4, s);
        if (s.error || chi_ref_write(state.reference, i * 64, words[i], '1))
          `uvm_fatal("SOURCE_IDS", "Initial source transaction failed")
      end
      for (int i = 0; i < 2; i++) begin
        r[i] = request('h1d, 3990 + i, i * 64);
        r[i].src = 3;
        send_req(r[i]);
        response(r[i], 6, dbid[i]);
      end
      repeat (4) @(negedge vif.clock);
      for (int i = 0; i < 8; i++) begin
        write_data(r[i], dbid[i], words[i], '1);
        response(r[i], 4, s);
        if (s.error || chi_ref_write(state.reference, i * 64, words[i], '1))
          `uvm_fatal("SOURCE_IDS", "Concurrent source transaction failed")
      end
      `uvm_info("PHASE", "Reset cancellation", UVM_LOW)
      // Cancel a write waiting for payload. SRAM data from completed writes survives.
      send_req(request('h1d, 110, 64));
      response(request('h1d, 110, 64), 6, s);
      reset_link();
      read_line(111, 64);
      write_line(110, 64, '1);
      write_line(110, 64, '1);  // Reuse is legal only after the first completion.
      read_line(112, 64);
      repeat (15) @(negedge vif.clock);
      if (state.sampled.outstanding != 0 || state.responses.size() || state.data_responses.size())
        `uvm_fatal("DRAIN", "Unmatched transactions remain")
      // Payload is unconstrained while FLITV=0 and must not create a transaction.
      @(negedge vif.clock);
      state.idle.req_flit = '0;
      state.idle.dat_flit = '0;
      repeat (2) @(negedge vif.clock);
      state.idle.req_flit = '1;
      state.idle.dat_flit = '1;
      repeat (2) @(negedge vif.clock);
      state.idle.req_flit = '0;
      state.idle.dat_flit = '0;
      repeat (2) @(negedge vif.clock);
      if (state.sampled.outstanding || state.responses.size() || state.data_responses.size())
        `uvm_fatal("INVALID_CHANNEL", "Inactive physical payload changed transactions")
      `uvm_info("CHECKS", $sformatf("CHI SN-F: %0d checked responses/data beats", state.checks),
                UVM_LOW)

    end
  endtask
  task body();
    scenarios();
  endtask
endclass
