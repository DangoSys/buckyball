package chi_retry_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  `include "flit_types.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual chi_retry_if vif;
    req_t requests[$];
    rsp_t responses[$], original_responses[$];
    int checks = 0;
    req_t originals[int];
    int retry_types[int];
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual chi_retry_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "Retry interface missing")
    endfunction
    task monitor();
      forever begin
        @(posedge vif.clock);
        if (vif.reset) begin
          requests.delete();
          responses.delete();
          original_responses.delete();
        end else begin
          if (vif.req_out_valid && vif.req_out_ready) requests.push_back(req_t'(vif.req_out));
          if (vif.rsp_out_valid && vif.rsp_out_ready) responses.push_back(rsp_t'(vif.rsp_out));
        end
      end
    endtask
    task send_request(req_t r);
      @(negedge vif.clock);
      vif.req_in = r;
      vif.req_in_valid = 1;
      do @(posedge vif.clock); while (!vif.req_in_ready);
      originals[r.txn] = r;
      @(negedge vif.clock);
      vif.req_in_valid = 0;
    endtask
    function bit coherent_read(int opcode);
      return opcode == 1 || opcode == 'h26 || opcode == 7;
    endfunction
    function bit coherent_request(int opcode);
      return coherent_read(opcode) || opcode == 'hd || opcode == 'h1b || opcode == 8 ||
          opcode == 9 || opcode == 10;
    endfunction
    task request(int txn, int opcode = 'hd);
      req_t r = '0;
      r.src = 1;
      r.tgt = 64;
      r.txn = txn;
      r.opcode = opcode;
      r.size = 6;
      r.addr = 64 * txn;
      r.allow_retry = 1;
      if (coherent_request(opcode)) begin
        r.mem_attr = 12;
        r.snp_attr = 1;
      end else begin
        r.mem_attr   = 4;
        r.return_nid = 1;
        r.return_txn = txn;
      end
      r.exp_comp_ack = coherent_read(opcode);
      send_request(r);
    endtask
    task response(int opcode, int txn, int src = 65, int credit_type = 3, bit contiguous = 0);
      rsp_t r = '0;
      r.opcode = opcode;
      r.txn = txn;
      r.src = src;
      r.tgt = 1;
      r.pcrd_type = (opcode == 3 || opcode == 7) ? credit_type : 0;
      r.qos = (txn ^ src) % 16;
      r.trace_tag = txn % 2;
      if (opcode == 6 || opcode == 5) r.dbid = (txn ^ src ^ 'h555) & 'hfff;
      if (opcode == 4 || opcode == 6 || opcode == 5) r.busy = (txn + src) % 8;
      if (opcode == 4 && txn % 2) r.error = 3;  // Dataless completion may report NDERR.
      if (!contiguous) @(negedge vif.clock);
      vif.rsp_in = r;
      vif.rsp_in_valid = 1;
      do @(posedge vif.clock); while (!vif.rsp_in_ready);
      if (opcode == 3) retry_types[txn] = credit_type;
      if (opcode != 3 && opcode != 7) original_responses.push_back(r);
      @(negedge vif.clock);
      vif.rsp_in_valid = 0;
    endtask
    task expect_request(int txn, bit retry);
      req_t r, golden;
      while (!requests.size()) @(negedge vif.clock);
      r = requests.pop_front();
      if (!originals.exists(txn)) `uvm_fatal("MODEL", "Missing observed original request")
      golden = originals[txn];
      if (retry) begin
        golden.allow_retry = 0;
        golden.pcrd_type   = retry_types[txn];
      end
      if (r !== golden)
        `uvm_fatal("RETRY", $sformatf(
                   "Complete request changed txn=%0h got=%h expected=%h", txn, r, golden))
      checks++;
    endtask
    task receive_response(output rsp_t r);
      rsp_t golden;
      while (!responses.size()) @(negedge vif.clock);
      if (!original_responses.size()) `uvm_fatal("MODEL", "Missing response/original")
      r = responses.pop_front();
      golden = original_responses.pop_front();
      if (r !== golden) `uvm_fatal("RESPONSE", "Complete response metadata changed")
    endtask
    task accept_data(int txn, int source);
      @(negedge vif.clock);
      vif.accepted_valid = 1;
      vif.accepted_src   = source;
      vif.accepted_txn   = txn;
      @(negedge vif.clock);
      vif.accepted_valid = 0;
    endtask
    task execute();
      rsp_t completed;
      vif.reset = 1;
      vif.req_in_valid = 0;
      vif.rsp_in_valid = 0;
      vif.accepted_valid = 0;
      vif.req_in = 0;
      vif.rsp_in = 0;
      vif.accepted_src = 0;
      vif.accepted_txn = 0;
      vif.req_out_ready = 1;
      vif.rsp_out_ready = 1;
      fork
        monitor();
      join_none
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
`ifdef CHI_RETRY_DUPLICATE
      request(12'hf01);
      expect_request(12'hf01, 0);
      request(12'hf01);
      `uvm_fatal("NEGATIVE", "Duplicate live TxnID did not terminate RTL")
`elsif CHI_RETRY_BAD_OPCODE
      request(0, 'h11);  // Separate response/data acceptance is outside this profile.
      `uvm_fatal("NEGATIVE", "Unsupported retry opcode did not terminate RTL")
`else
      // A normal completion passes through and releases the request record.
      request(0);
      expect_request(0, 0);
      vif.rsp_out_ready = 0;
      fork
        response(4, 0);
        begin
          repeat (12) @(negedge vif.clock);
          vif.rsp_out_ready = 1;
        end
      join
      receive_response(completed);
      if (completed.opcode != 4) `uvm_fatal("PASS", "Completion was swallowed")
      for (int i = 0; i < 4; i++) begin
        request(i);
        expect_request(i, 0);
      end
      for (int i = 0; i < 4; i++) response(3, i);
      repeat (10) @(negedge vif.clock);
      if (requests.size() || responses.size() || original_responses.size() || vif.pending != 4)
        `uvm_fatal("WAIT", "Retry issued before P-Credit or internal response leaked")
      // Home remapped from TgtID 64 to RetryAck SrcID 65; match that credit source.
      response(7, 0, 66, 3);
      response(7, 0, 65, 2);
      repeat (10) @(negedge vif.clock);
      if (requests.size()) `uvm_fatal("MATCH", "Mismatched protocol credit consumed")
      for (int i = 0; i < 4; i++) begin
        response(7, 0);
        expect_request(i, 1);
        response(4, i);
        receive_response(completed);
        if (completed.txn != i) `uvm_fatal("COMP", "Bad completion");
      end
      // An output stall must preserve the queued request and its record.
      vif.req_out_ready = 0;
      request(0);
      repeat (12) @(negedge vif.clock);
      if (!vif.req_out_valid || requests.size())
        `uvm_fatal("STALL", "Queued request lost under stall")
      vif.req_out_ready = 1;
      expect_request(0, 0);
      response(4, 0);
      receive_response(completed);
      if (completed.txn != 0) `uvm_fatal("COMP", "Bad stalled completion");
      // Grant can precede RetryAck and must be retained for its matching retry.
      request(0, 4);
      expect_request(0, 0);
      response(7, 0);
      response(3, 0);
      expect_request(0, 1);
      @(negedge vif.clock);
      vif.accepted_valid = 1;
      vif.accepted_src   = 9;
      vif.accepted_txn   = 0;
      @(negedge vif.clock);
      vif.accepted_valid = 0;
      repeat (4) @(negedge vif.clock);
      if (vif.pending != 0 || requests.size() || responses.size() || original_responses.size())
        `uvm_fatal("DRAIN", "Retry records did not drain")
      // Reset cancels retry-wait records and retained protocol credits.
      request(0);
      expect_request(0, 0);
      response(3, 0);
      @(negedge vif.clock);
      vif.reset = 1;
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      request(0);
      expect_request(0, 0);
      response(4, 0);
      receive_response(completed);
      if (completed.txn != 0) `uvm_fatal("RESET", "Completion after reset corrupted");
      repeat (4) @(negedge vif.clock);
      if (vif.pending != 0 || requests.size() || responses.size() || original_responses.size())
        `uvm_fatal("DRAIN", "Retry reset did not drain")
      // Full protocol IDs are independent of the four local acceptance slots.
      // Exercise both polarities of every aligned address, TxnID and routing-ID bit.
      for (int phase_index = 0; phase_index < 3; phase_index++) begin
        int polarity = phase_index % 2;
        for (int address_bit = 6; address_bit < 44; address_bit++) begin
          int ids[4], homes[4], types[4], opcodes[4];
          bit [3:0] seen_reissues = '0;
          int supported_ops[11] = '{1, 'h26, 7, 'hd, 'h1b, 8, 9, 10, 4, 'h1d, 'h1c};
          for (int slot = 0; slot < 4; slot++) begin
            req_t r = '0;
            int   node_bits = (1 << ((address_bit + slot) % 7)) | 2;
            ids[slot] = (1 << ((address_bit + slot) % 12)) ^ (polarity ? 'hfff : 0);
            homes[slot] = (address_bit % 3 == 0 ? 65 : node_bits ^ (polarity ? 'h7f : 0));
            types[slot] = (address_bit % 3 == 1 ? address_bit % 16 : (address_bit + slot) % 16);
            r.src = 1;
            r.tgt = homes[slot];
            r.txn = ids[slot];
            opcodes[slot] = supported_ops[(address_bit+slot+polarity)%11];
            r.opcode = opcodes[slot];
            r.size = 6;
            r.lpid = (address_bit + slot) % 32;
            r.likely_shared = (address_bit + slot) % 2;
            if (coherent_request(opcodes[slot])) begin
              r.mem_attr = 12;
              r.snp_attr = 1;
            end else begin
              case ((address_bit + slot) % 3)
                0: r.mem_attr = 0;
                1: r.mem_attr = 4;
                2: r.mem_attr = 12;
              endcase
              r.mem_attr |= (address_bit + slot + polarity) % 2;  // EWA is legal for Normal memory.
              r.return_nid = homes[slot];
              r.return_txn = ids[slot] ^ 'hfff;
            end
            r.exp_comp_ack = coherent_read(opcodes[slot]);
            r.addr = ((64'h1 << address_bit) ^ (polarity ? 64'hfffffffffc0 : 0));
            r.qos = (address_bit + 3 * slot) % 16;
            r.trace_tag = polarity;
            r.allow_retry = 1;
            send_request(r);
            expect_request(ids[slot], 0);
          end
          if (vif.pending != 4)
            `uvm_fatal("SLOTS", "Arbitrary IDs did not fill four acceptance slots")
          vif.rsp_out_ready = 0;  // Internal retry traffic must bypass endpoint backpressure.
          for (int position = 0; position < 4; position++) begin
            int slot = (position + address_bit / 4) % 4;
            response(7, 0, homes[slot], types[slot]);
          end
          if (address_bit % 5 == 0) vif.req_out_ready = 0;
          for (int position = 0; position < 4; position++) begin
            int slot = (address_bit + (address_bit % 2 ? 3 - position : position)) % 4;
            response(3, ids[slot], homes[slot], types[slot], position != 0);
          end
          repeat (3) @(negedge vif.clock);
          vif.req_out_ready = 1;
          vif.rsp_out_ready = 1;
          for (int position = 0; position < 4; position++) begin
            int slot = -1;
            while (!requests.size()) @(negedge vif.clock);
            for (int candidate = 0; candidate < 4; candidate++)
            if (requests[0].txn == ids[candidate]) slot = candidate;
            if (slot < 0 || seen_reissues[slot])
              `uvm_fatal("RETRY_SET", "Unexpected or duplicate reissued transaction")
            seen_reissues[slot] = 1;
            expect_request(ids[slot], 1);
            if (coherent_read(opcodes[slot]) || opcodes[slot] == 4) begin
              accept_data(ids[slot], homes[slot]);
            end else begin
              int acceptance=(opcodes[slot]=='h1b || opcodes[slot]=='h1d || opcodes[slot]=='h1c) ? 6 : 4;
              if (acceptance == 6 && (address_bit + slot) % 2) acceptance = 5;
              response(acceptance, ids[slot], homes[slot], 0);
              receive_response(completed);
              if (completed.txn != ids[slot] || completed.opcode != acceptance)
                `uvm_fatal("PASS", "Acceptance response corrupted")
            end
          end
          repeat (3) @(negedge vif.clock);
          if (vif.pending != 0 || requests.size() || responses.size() || original_responses.size())
            `uvm_fatal("DRAIN", "Full-width request batch did not drain")
        end
      end
      // Fill the two-entry output queue and hold a third request under backpressure.
      vif.req_out_ready = 0;
      fork
        begin
          for (int slot = 0; slot < 4; slot++) request(slot);
        end
        begin
          repeat (20) @(negedge vif.clock);
          vif.req_out_ready = 1;
        end
      join
      for (int slot = 0; slot < 4; slot++) expect_request(slot, 0);
      for (int slot = 0; slot < 4; slot++) begin
        response(4, slot);
        receive_response(completed);
      end
      // Initial allocation overlaps independent RetryAck and completion handshakes.
      for (int context_kind = 0; context_kind < 2; context_kind++) begin
        for (int slot = 0; slot < 4; slot++) begin
          int other = (slot + 1) % 4;
          for (int i = 0; i < 4; i++) begin
            request(100 + i);
            expect_request(100 + i, 0);
          end
          response(4, 100 + slot);
          receive_response(completed);
          fork
            request(200 + slot);
            response(context_kind ? 4 : 3, 100 + other);
          join
          expect_request(200 + slot, 0);
          if (!context_kind) begin
            response(7, 0);
            expect_request(100 + other, 1);
          end else begin
            receive_response(completed);
          end
          for (int i = 0; i < 4; i++) begin
            if (context_kind && i == other) continue;
            response(4, i == slot ? 200 + slot : 100 + i);
            receive_response(completed);
          end
        end
      end
      // A new P-Credit may arrive in the cycle a different credit is consumed.
      for (int slot = 0; slot < 4; slot++) begin
        int next_slot = (slot + 1) % 4;
        for (int i = 0; i < 4; i++) begin
          request(300 + i);
          expect_request(300 + i, 0);
        end
        response(7, 0);
        response(3, 300 + slot);
        response(7, 0, 65, 3, 1);
        expect_request(300 + slot, 1);
        response(3, 300 + next_slot);
        expect_request(300 + next_slot, 1);
        for (int i = 0; i < 4; i++) begin
          response(4, 300 + i);
          receive_response(completed);
        end
      end
      // Later split completion/data acceptance may follow record release.
      accept_data(3, 65);
      repeat (3) @(negedge vif.clock);
      if (vif.pending || requests.size() || responses.size() || original_responses.size())
        `uvm_fatal("DRAIN", "Saturated output queue did not drain")
      // Acceptance tracking can retire before later data beats or split completion.
      request(500, 4);
      expect_request(500, 0);
      accept_data(500, 9);
      request(501, 'h1d);
      expect_request(501, 0);
      response(6, 501);
      receive_response(completed);
      for (int i = 0; i < 4; i++) begin
        request(600 + i);
        expect_request(600 + i, 0);
      end
      for (int i = 0; i < 4; i++) response(3, 600 + i);
      for (int i = 0; i < 4; i++) begin
        rsp_t idle = '0;
        idle.opcode = 4;
        idle.tgt = 1;
        idle.txn = 600 + i;
        @(negedge vif.clock);
        vif.rsp_in = idle;
        vif.accepted_txn = 600 + i;
        @(negedge vif.clock);
        if (vif.pending != 4 || requests.size() || responses.size())
          `uvm_fatal("INVALID_CHANNEL", "Inactive matching payload changed a retry-wait record")
      end
      response(4, 501);
      receive_response(completed);
      accept_data(500, 9);
      if (vif.pending != 4) `uvm_fatal("LATE_ACCEPT", "Late acceptance retired an unrelated retry")
      for (int i = 0; i < 4; i++) begin
        response(7, 0);
        expect_request(600 + i, 1);
        response(4, 600 + i);
        receive_response(completed);
      end
      // VALID held through coordinated reset is accepted once after release.
      begin
        req_t held = '0;
        held.src = 1;
        held.tgt = 64;
        held.txn = 'ha55;
        held.opcode = 'hd;
        held.size = 6;
        held.addr = 64;
        held.allow_retry = 1;
        held.mem_attr = 12;
        held.snp_attr = 1;
        @(negedge vif.clock);
        vif.reset = 1;
        vif.req_in = held;
        vif.req_in_valid = 1;
        repeat (4) @(negedge vif.clock);
        vif.reset = 0;
        do @(posedge vif.clock); while (!vif.req_in_ready);
        originals[held.txn] = held;
        @(negedge vif.clock);
        vif.req_in_valid = 0;
        expect_request(held.txn, 0);
        response(4, held.txn);
        receive_response(completed);
        repeat (3) @(negedge vif.clock);
        if (vif.pending || requests.size() || responses.size() || original_responses.size())
          `uvm_fatal("RESET", "Held VALID did not complete exactly once")
      end
      // Invalid channels may carry arbitrary payload; no transaction may be consumed.
      begin
        req_t idle_req = '0;
        rsp_t idle_rsp = '0;
        idle_req.src = 1;
        idle_req.opcode = 'hd;
        idle_req.size = 6;
        idle_req.mem_attr = 12;
        idle_req.snp_attr = 1;
        idle_req.allow_retry = 1;
        idle_rsp.tgt = 1;
        idle_rsp.opcode = 4;
        for (int b = 0; b < $bits(req_t); b++) begin
          @(negedge vif.clock);
          vif.req_in = idle_req ^ (137'b1 << b);
          @(negedge vif.clock);
          vif.req_in = idle_req;
          if (vif.pending || vif.req_out_valid || vif.rsp_out_valid)
            `uvm_fatal("INVALID_CHANNEL", "Invalid REQ payload changed live transactions")
        end
        @(negedge vif.clock);
        idle_req.snp_attr = 0;
        idle_req.mem_attr = 2;
        vif.req_in = idle_req;
        @(negedge vif.clock);
        if (vif.pending || vif.req_out_valid)
          `uvm_fatal("INVALID_CHANNEL", "Inactive Device payload was consumed")
        for (int b = 0; b < $bits(rsp_t); b++) begin
          @(negedge vif.clock);
          vif.rsp_in = idle_rsp ^ (71'b1 << b);
          @(negedge vif.clock);
          vif.rsp_in = idle_rsp;
          if (vif.pending || vif.req_out_valid || vif.rsp_out_valid)
            `uvm_fatal("INVALID_CHANNEL", "Invalid RSP payload changed live transactions")
        end
      end
      repeat (2) @(negedge vif.clock);
      `uvm_info("CHECKS", $sformatf("CHI retry: %0d request/reissue fields checked", checks),
                UVM_LOW)
`endif
    endtask
  endclass
endpackage
