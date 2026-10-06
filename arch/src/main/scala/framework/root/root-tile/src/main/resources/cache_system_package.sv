`include "cache_system_config.svh"
`define CSF(K, F) `COH_``K``_``F``_OFFSET +: `COH_``K``_``F``_WIDTH
package cache_system_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  typedef logic [`COH_REQ_WIDTH-1:0] req_t;
  typedef logic [`COH_RSP_WIDTH-1:0] rsp_t;
  typedef logic [`COH_DAT_WIDTH-1:0] dat_t;
  typedef bit [`COH_MEMRESP_WIDTH-1:0] mem_t;
  import "DPI-C" function chandle coherence_ref_create();
  import "DPI-C" function void coherence_ref_destroy(input chandle model);
  import "DPI-C" function void coherence_ref_read(
    input chandle model,
    input longint unsigned addr,
    output bit [511:0] data
  );
  import "DPI-C" function void coherence_ref_write(
    input chandle model,
    input longint unsigned addr,
    input bit [511:0] data
  );
  class result_item extends uvm_sequence_item;
    bit [63:0] data;
    bit error;
    `uvm_object_utils_begin(result_item)
      `uvm_field_int(data, UVM_DEFAULT)
      `uvm_field_int(error, UVM_DEFAULT)
    `uvm_object_utils_end
    function new(string name = "result_item");
      super.new(name);
    endfunction
  endclass
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual cache_system_if vif;
    virtual stream_if #(`COH_MEMREQ_WIDTH) mem_req;
    virtual stream_if #(`COH_MEMRESP_WIDTH) mem_resp;
    in_order_scoreboard #(result_item) scoreboard[2];
    chandle architectural, backing;
    typedef struct packed {
      bit write;
      bit [63:0] addr, data;
      bit [7:0] mask;
    } command_t;
    typedef command_t command_queue[$];
    command_queue commands[2];
    typedef struct {
      mem_t bits;
      int   due,  serial;
    } memory_entry;
    memory_entry pending[$];
    req_t requests[int];
    bit [511:0] read_expected[int], snoop_expected[int];
    longint unsigned wb_address[int];
    int
        fill_seen[int],
        fill_dbid[int],
        snoop_seen[int],
        snoop_target[int],
        wb_seen[int],
        wb_owner[int],
        slot_owner[int];
    bit ack_pending[int];
    int submitted[2] = '{0, 0};
    int result_stalls[2] = '{0, 0};
    int home_requests = 0, rsp_block_cycles = 0, release_cycle = -1, ack_resume_cycles = -1;
    int
        cycle = 0,
        peak = 0,
        memory_reads = 0,
        memory_writes = 0,
        memory_serial = 0,
        last_memory_serial = -1;
    int full_lines = 0, snoop_lines = 0, copybacks = 0, dirty_snoops = 0, acks = 0, errors = 0;
    bit
        hold_memory = 0,
        hold_results = 0,
        mem_active = 0,
        saw_reorder = 0,
        saw_dbid_mismatch = 0,
        saw_large_snoop = 0;
    bit [`COH_MEMRESP_WIDTH-1:0] mem_packet;
    localparam bit [63:0] ERROR_ADDRESS = 64'hf000;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 1ms;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if(`COH_AGENTS!=2||`COH_MSHRS!=4||`COH_SETS!=4||`COH_WAYS!=2||`COH_DATA_BITS!=256)
        `uvm_fatal(
            "CONFIG",
            "CacheSystem regression requires 2 requesters, 4 MSHRs/sets, 2 ways and 256-bit DAT")
      if (!uvm_config_db#(virtual cache_system_if)::get(
              this, "", "control", vif
          ) || !uvm_config_db#(virtual stream_if #(`COH_MEMREQ_WIDTH))::get(
              this, "", "mem_req", mem_req
          ) || !uvm_config_db#(virtual stream_if #(`COH_MEMRESP_WIDTH))::get(
              this, "", "mem_resp", mem_resp
          ))
        `uvm_fatal("VIF", "CacheSystem interfaces missing")
      for (int i = 0; i < 2; i++)
      scoreboard[i] =
          in_order_scoreboard#(result_item)::type_id::create($sformatf("scoreboard_%0d", i), this);
      architectural = coherence_ref_create();
      backing = coherence_ref_create();
    endfunction
    function void observe_cpu();
      for (int core = 0; core < 2; core++) begin
        if (vif.sample.access_valid[core] && vif.sample.access_ready[core]) begin
          command_t cmd;
          bit [511:0] line;
          result_item expected;
          cmd = '{
              vif.sample.access_write[core],
              64'(vif.sample.access_addr[core]),
              vif.sample.access_data[core],
              vif.sample.access_mask[core]
          };
          if (vif.sample.access_atomicWord[core] !== 1'b0)
            `uvm_fatal("PROFILE", "This regression drives atomicWord=False")
          commands[core].push_back(cmd);
          coherence_ref_read(architectural, cmd.addr & ~64'h3f, line);
          expected = new;
          expected.error = cmd.addr == ERROR_ADDRESS;
          expected.data = expected.error ? 0 : line[int'(cmd.addr&63)*8+:64];
          scoreboard[core].expected_export.write(expected);
          submitted[core]++;
        end
        if (vif.sample.result_valid[core] && vif.sample.result_ready[core]) begin
          command_t cmd;
          bit [511:0] line;
          result_item actual;
          if ($isunknown({vif.sample.result_data[core], vif.sample.result_error[core]}))
            `uvm_fatal("RESULT_X", "Unknown CPU result")
          if (!commands[core].size())
            `uvm_fatal("EXTRA_RESULT", "CPU result without an accepted command")
          cmd = commands[core].pop_front();
          actual = new;
          actual.data = vif.sample.result_data[core];
          actual.error = vif.sample.result_error[core];
          scoreboard[core].actual_export.write(actual);
          if (actual.error) errors++;
          if (cmd.write && !actual.error) begin
            coherence_ref_read(architectural, cmd.addr & ~64'h3f, line);
            for (int b = 0; b < 8; b++)
            if (cmd.mask[b]) line[(int'(cmd.addr&63)+b)*8+:8] = cmd.data[b*8+:8];
            coherence_ref_write(architectural, cmd.addr & ~64'h3f, line);
          end
        end
      end
    endfunction
    function void observe_protocol();
      if (vif.sample.req_valid && $isunknown(vif.sample.req_bits))
        `uvm_fatal("REQ_X", "Unknown accepted RequestFlit")
      if (vif.sample.rsp_valid && $isunknown(vif.sample.rsp_bits))
        `uvm_fatal("RSP_X", "Unknown accepted ResponseFlit")
      if (vif.sample.dat_valid && $isunknown(vif.sample.dat_bits))
        `uvm_fatal("DAT_X", "Unknown accepted DataFlit")
      if (vif.sample.snp_valid && $isunknown(vif.sample.snp_bits))
        `uvm_fatal("SNP_X", "Unknown accepted DirectedSnoop")
      if (vif.sample.rx_rsp_valid && $isunknown(vif.sample.rx_rsp_bits))
        `uvm_fatal("RX_RSP_X", "Unknown requester response")
      if (vif.sample.rx_dat_valid && $isunknown(vif.sample.rx_dat_bits))
        `uvm_fatal("RX_DAT_X", "Unknown requester data")
      if (vif.sample.req_valid) begin
        req_t r;
        int   key;
        home_requests++;
        r   = vif.sample.req_bits;
        key = (int'(r[`CSF(REQ, SRCID)]) << 1) | int'(r[`CSF(REQ, TXNID)]);
        if (r[
            `CSF(REQ, TGTID)
            ] != 64 || r[
            `CSF(REQ, SRCID)
            ] < 1 || r[
            `CSF(REQ, SRCID)
            ] > 2 || r[
            `CSF(REQ, TXNID)
            ] >= 2)
          `uvm_fatal("REQ_ID", $sformatf(
                     "Requester identity or bank TxnID corrupted src=%0d tgt=%0d txn=%0d packet=%h",
                     r[
                     `CSF(REQ, SRCID)
                     ],
                     r[
                     `CSF(REQ, TGTID)
                     ],
                     r[
                     `CSF(REQ, TXNID)
                     ],
                     r
                     ))
        requests[key] = r;
        if (r[`CSF(REQ, OPCODE)] == `COH_READ_UNIQUE || r[`CSF(REQ, OPCODE)] == `COH_READ_NSD) begin
          coherence_ref_read(architectural, r[`CSF(REQ, ADDR)], read_expected[key]);
          fill_seen[key] = 0;
        end
      end
      if (vif.sample.rsp_valid) begin
        rsp_t r;
        int key, id;
        r   = vif.sample.rsp_bits;
        key = (int'(r[`CSF(RSP, TGTID)]) << 1) | int'(r[`CSF(RSP, TXNID)]);
        id  = r[`CSF(RSP, DBID)];
        if (!requests.exists(key) || r[`CSF(RSP, SRCID)] != 64 || r[`CSF(RSP, RESPERR)] != 0)
          `uvm_fatal("RSP_ID", "Unexpected completion identity or error")
        if (r[`CSF(RSP, OPCODE)] == `COH_COMP_DBID) begin
          if (id >= 4 || requests[key][`CSF(REQ, OPCODE)] != `COH_WRITEBACK)
            `uvm_fatal("WB_ID", "Writeback grant does not match request")
          wb_address[id] = requests[key][`CSF(REQ, ADDR)];
          wb_owner[id] = r[`CSF(RSP, TGTID)];
          wb_seen[id] = 0;
        end else if (r[`CSF(RSP, OPCODE)] != `COH_COMP)
          `uvm_fatal("RETRY", "Canonical Home profile must backpressure REQ, not generate Retry")
      end
      if (vif.sample.dat_valid) begin
        dat_t d;
        int key, b, id;
        bit failed;
        bit [255:0] expected;
        d = vif.sample.dat_bits;
        key = (int'(d[`CSF(DAT, TGTID)]) << 1) | int'(d[`CSF(DAT, TXNID)]);
        id = d[`CSF(DAT, DBID)];
        b = int'(d[`CSF(DAT, DATAID)]) >> 1;
        failed = d[`CSF(DAT, RESPERR)] != 0;
        if (!requests.exists(
                key
            ) || !read_expected.exists(
                key
            ) || id >= 4 || d[
            `CSF(DAT, SRCID)
            ] != 64 || d[
            `CSF(DAT, HOMENID)
            ] != 64 || d[
            `CSF(DAT, OPCODE)
            ] != `COH_COMP_DATA)
          `uvm_fatal("DAT_ID", "CompData bank TxnID, HomeNID or DBID mismatch")
        if (d[`CSF(DAT, DATAID)] != 0 && d[`CSF(DAT, DATAID)] != 2)
          `uvm_fatal("DATAID", "Invalid 256-bit DataID")
        if (fill_seen[key] & (1 << b)) `uvm_fatal("DUPLICATE", "Duplicate CompData beat")
        if (fill_seen[key] && fill_dbid[key] != id)
          `uvm_fatal("DBID", "CompData DBID changed within line")
        if (failed != (requests[key][`CSF(REQ, ADDR)] == ERROR_ADDRESS))
          `uvm_fatal("ERROR", "Unexpected backend error status")
        expected = failed ? '0 : read_expected[key][b*256+:256];
        if (d[`CSF(DAT, DATA)] !== expected || d[`CSF(DAT, BE)] !== (failed ? 32'h0 : 32'hffffffff))
          `uvm_fatal("LINE_DATA", "Full 64-byte CompData payload/BE mismatch")
        if (failed && d[`CSF(DAT, RESPERR)] != 3) `uvm_fatal("NDERR", "Read fault was not NDERR")
        if (!fill_seen[key]) begin
          fill_dbid[key] = id;
          slot_owner[id] = key;
        end
        fill_seen[key] |= 1 << b;
        if (id != d[`CSF(DAT, TXNID)]) saw_dbid_mismatch = 1;
        if (fill_seen[key] == 3) begin
          full_lines++;
          ack_pending[id] = 1;
        end
      end
      if (vif.sample.snp_valid) begin
        int id, target;
        bit [63:0] address;
        id = vif.sample.snp_bits[`CSF(SNP, TXNID)];
        target = vif.sample.snp_bits[`CSF(SNP, TARGET)];
        address = 64'(vif.sample.snp_bits[`CSF(SNP, ADDR)]) << 3;
        if (id >= 4 || target < 1 || target > 2 || vif.sample.snp_bits[`CSF(SNP, SRCID)] != 64)
          `uvm_fatal("SNP_ID", "Directed snoop lost target/slot/address semantics")
        coherence_ref_read(architectural, address, snoop_expected[id]);
        snoop_seen[id]   = 0;
        snoop_target[id] = target;
        if (id >= 2) saw_large_snoop = 1;
      end
      if (vif.sample.rx_rsp_valid) begin
        rsp_t r;
        int   id;
        r  = vif.sample.rx_rsp_bits;
        id = r[`CSF(RSP, TXNID)];
        if (r[`CSF(RSP, OPCODE)] == `COH_COMP_ACK) begin
          if (!ack_pending.exists(
                  id
              ) || !ack_pending[id] || slot_owner[id] >> 1 != r[
              `CSF(RSP, SRCID)
              ] || r[
              `CSF(RSP, TGTID)
              ] != 64)
            `uvm_fatal("ACK_ID", "CompAck must return the Home DBID, not original bank TxnID")
          ack_pending[id] = 0;
          acks++;
        end else begin
          if (r[
              `CSF(RSP, OPCODE)
              ] != `COH_SNP_RESP || !snoop_target.exists(
                  id
              ) || r[
              `CSF(RSP, SRCID)
              ] != snoop_target[id] || r[
              `CSF(RSP, TGTID)
              ] != 64)
            `uvm_fatal("RX_RSP", "Snoop response identity mismatch")
        end
      end
      if (vif.sample.rx_dat_valid) begin
        dat_t d;
        int id, b;
        bit [511:0] expected;
        d  = vif.sample.rx_dat_bits;
        id = d[`CSF(DAT, TXNID)];
        b  = int'(d[`CSF(DAT, DATAID)]) >> 1;
        if (id >= 4 || (d[`CSF(DAT, DATAID)] != 0 && d[`CSF(DAT, DATAID)] != 2))
          `uvm_fatal("RETURN_ID", "Snoop/writeback must return full Home slot ID")
        if (d[`CSF(DAT, OPCODE)] == `COH_SNP_DATA) begin
          if (!snoop_expected.exists(
                  id
              ) || d[
              `CSF(DAT, SRCID)
              ] != snoop_target[id] || d[
              `CSF(DAT, TGTID)
              ] != 64 || (snoop_seen[id] & (1 << b)))
            `uvm_fatal("SNP_DATA", "Unmatched or duplicate snoop data")
          expected = snoop_expected[id];
          snoop_seen[id] |= 1 << b;
          if (snoop_seen[id] == 3) begin
            snoop_lines++;
            if (d[`CSF(DAT, RESP)] & 4) dirty_snoops++;
          end
        end else if (d[`CSF(DAT, OPCODE)] == `COH_COPYBACK_DATA) begin
          if (!wb_address.exists(
                  id
              ) || d[
              `CSF(DAT, SRCID)
              ] != wb_owner[id] || d[
              `CSF(DAT, TGTID)
              ] != 64 || (wb_seen[id] & (1 << b)))
            `uvm_fatal("WB_DATA", "Unmatched or duplicate copyback")
          coherence_ref_read(architectural, wb_address[id], expected);
          wb_seen[id] |= 1 << b;
          if (wb_seen[id] == 3) copybacks++;
          if (d[`CSF(DAT, RESP)] == 0) expected = '0;
        end else `uvm_fatal("RETURN_OPCODE", "Unexpected requester data opcode")
        if (d[`CSF(DAT, DATA)] !== expected[b*256+:256])
          `uvm_fatal("RETURN_PAYLOAD", "Full-line snoop/copyback data mismatch")
      end
    endfunction
    task service();
      forever begin
        @(vif.sample);
        if (!vif.sample.reset) begin
          cycle++;
          if (vif.sample.outstanding > peak) peak = vif.sample.outstanding;
          observe_cpu();
          observe_protocol();
          for (int core = 0; core < 2; core++)
          if (vif.sample.result_valid[core] && !vif.sample.result_ready[core])
            result_stalls[core]++;
          if (vif.sample.block_requester_rsp) begin
            rsp_block_cycles++;
            if (!vif.sample.active || vif.sample.rx_rsp_valid)
              `uvm_fatal("RSP_BLOCK", "RSP stall changed active or allowed a blocked response")
          end
          if (release_cycle >= 0 && ack_resume_cycles < 0 && acks >= 4)
            ack_resume_cycles = cycle - release_cycle;
          if (mem_active && mem_resp.sample.valid && mem_resp.sample.ready) mem_active = 0;
          if (mem_req.sample.valid && mem_req.sample.ready) begin
            memory_entry entry;
            bit [511:0] line, expected;
            bit write;
            longint unsigned addr;
            if ($isunknown(
                    {
                      mem_req.sample.bits[`CSF(MEMREQ, ID)],
                      mem_req.sample.bits[`CSF(MEMREQ, ADDR)],
                      mem_req.sample.bits[`CSF(MEMREQ, WRITE)]
                    }
                ))
              `uvm_fatal("MEM_X", "Unknown backing command")
            entry.bits = '0;
            entry.bits[`CSF(MEMRESP, ID)] = mem_req.sample.bits[`CSF(MEMREQ, ID)];
            addr = mem_req.sample.bits[`CSF(MEMREQ, ADDR)];
            write = mem_req.sample.bits[`CSF(MEMREQ, WRITE)];
            if (addr[5:0] != 0) `uvm_fatal("MEM_ALIGN", "Backing request not line aligned")
            foreach (pending[i])
            if (pending[i].bits[`CSF(MEMRESP, ID)] == entry.bits[`CSF(MEMRESP, ID)])
              `uvm_fatal("MEM_ID", "Reused live backing ID")
            if (write) begin
              if ($isunknown(
                      {
                        mem_req.sample.bits[`CSF(MEMREQ, DATA)],
                        mem_req.sample.bits[`CSF(MEMREQ, MASK)]
                      }
                  ))
                `uvm_fatal("MEM_WRITE_X", "Unknown backing write payload")
              line = mem_req.sample.bits[`CSF(MEMREQ, DATA)];
              coherence_ref_read(architectural, addr, expected);
              if (mem_req.sample.bits[`CSF(MEMREQ, MASK)] !== '1 || line !== expected)
                `uvm_fatal("MEM_WRITE", "Backing write differs from architectural full line")
              coherence_ref_write(backing, addr, line);
              memory_writes++;
            end else begin
              coherence_ref_read(backing, addr, line);
              memory_reads++;
            end
            entry.bits[`CSF(MEMRESP, DATA)] = line;
            entry.bits[`CSF(MEMRESP, ERROR)] = !write && addr == ERROR_ADDRESS;
            entry.due = cycle + 9 + (3 - int'(entry.bits[`CSF(MEMRESP, ID)])) * 3;
            entry.serial = memory_serial++;
            pending.push_back(entry);
          end
        end
        @(negedge vif.clock);
        mem_req.ready = !vif.reset && cycle % 7 != 0;
        for (int core = 0; core < 2; core++)
        vif.result_ready[core] = !vif.reset && !hold_results && (cycle + core) % 11 >= 3;
        if (!mem_active && !hold_memory) begin
          for (int i = pending.size() - 1; i >= 0; i--)
          if (pending[i].due <= cycle) begin
            memory_entry entry;
            entry = pending[i];
            pending.delete(i);
            mem_packet = entry.bits;
            mem_active = 1;
            if (entry.serial < last_memory_serial) saw_reorder = 1;
            last_memory_serial = entry.serial;
            break;
          end
        end
        mem_resp.valid = mem_active;
        mem_resp.bits  = mem_packet;
      end
    endtask
    task send(int core, longint unsigned address, bit write = 0, bit [63:0] data = 0,
              bit [7:0] mask = 8'hff);
      @(negedge vif.clock);
      vif.access_addr[core] = address;
      vif.access_write[core] = write;
      vif.access_data[core] = data;
      vif.access_mask[core] = mask;
      vif.access_atomic[core] = 0;
      vif.access_atomicWord[core] = 0;
      vif.access_valid[core] = 1;
      do @(vif.sample); while (!vif.sample.access_ready[core]);
      @(negedge vif.clock);
      vif.access_valid[core] = 0;
    endtask
    task settle();
      for (int core = 0; core < 2; core++) scoreboard[core].wait_checked(submitted[core]);
      do @(vif.sample); while (vif.sample.outstanding != 0 || pending.size() != 0 || mem_active);
      repeat (8) @(vif.sample);
    endtask
    task execute();
      vif.reset = 1;
      vif.active = 0;
      vif.block_requester_rsp = 0;
      mem_req.ready = 0;
      mem_resp.valid = 0;
      mem_resp.bits = '0;
      for (int i = 0; i < 2; i++) begin
        vif.access_valid[i] = 0;
        vif.access_addr[i] = 0;
        vif.access_write[i] = 0;
        vif.access_data[i] = 0;
        vif.access_mask[i] = '1;
        vif.access_atomic[i] = 0;
        vif.access_atomicWord[i] = 0;
        vif.result_ready[i] = 0;
      end
      repeat (5) @(negedge vif.clock);
      vif.reset  = 0;
      vif.active = 1;
      fork
        service();
      join_none
      repeat (12) @(vif.sample);
      // Completion ACKs remain in the physical Rx FIFOs while DAT and CPU result
      // channels make progress. Hold CPU results first, then issue more REQs while
      // all four Home slots still wait for their ACKs.
      hold_memory  = 1;
      hold_results = 1;
      @(negedge vif.clock);
      vif.block_requester_rsp = 1;
      fork
        begin
          send(0, 'h0);
          send(0, 'h40);
        end
        begin
          send(1, 'h80);
          send(1, 'hc0);
        end
      join
      wait (pending.size() == 4);
      if (peak != 4)
        `uvm_fatal("CONCURRENCY", "Four independent misses did not occupy all Home MSHRs")
      hold_memory = 0;
      wait (vif.sample.result_valid[0] && vif.sample.result_valid[1]);
      repeat (24) begin
        @(vif.sample);
        if(vif.sample.outstanding!=4||acks!=0||scoreboard[0].checked!=0||scoreboard[1].checked!=0)
          `uvm_fatal("ACK_LIFETIME", "Blocked ACK or CPU result prematurely retired")
      end
      hold_results = 0;
      scoreboard[0].wait_checked(2);
      scoreboard[1].wait_checked(2);
      repeat (24) begin
        @(vif.sample);
        if (vif.sample.outstanding != 4 || acks != 0 || full_lines != 4)
          `uvm_fatal("ACK_LIFETIME", "Home released a slot before receiving CompAck")
      end
      fork
        send(0, 'h800);
        send(1, 'h880);
      join
      repeat (16) begin
        @(vif.sample);
        if(vif.sample.outstanding!=4||home_requests!=4||scoreboard[0].checked!=2||scoreboard[1].checked!=2)
          `uvm_fatal("REQ_PRESSURE", "New request advanced while all slots awaited blocked ACKs")
      end
      @(negedge vif.clock);
      release_cycle = cycle;
      vif.block_requester_rsp = 0;
      begin
        bit drained = 0;
        for (int elapsed = 0; elapsed < 256; elapsed++) begin
          @(vif.sample);
          if(scoreboard[0].checked==submitted[0]&&scoreboard[1].checked==submitted[1]&&
        vif.sample.outstanding==0&&pending.size()==0&&!mem_active)begin
            drained = 1;
            break;
          end
        end
        if(!drained||ack_resume_cycles<0||ack_resume_cycles>32||result_stalls[0]<20||result_stalls[1]<20)
          `uvm_fatal("ACK_PROGRESS",
                     "CompAck/blocked REQ/result channels did not drain within the progress bound")
      end
      settle();
      `uvm_info(
          "BACKPRESSURE",
          $sformatf(
              "4 MSHRs held for CompAck with live credit links; CPU stall cycles=%0d/%0d RSP blocked=%0d first4ACK resume=%0d cycles; pending REQs and all results drained",
              result_stalls[0], result_stalls[1], rsp_block_cycles, ack_resume_cycles), UVM_LOW)
      // Prime two RN1 lines, then keep slots0/1 waiting for unrelated backing reads.
      send(0, 'h80);
      settle();
      send(0, 'hc0);
      settle();
      hold_memory = 1;
      send(0, 'h400);
      send(0, 'h440);
      wait (pending.size() == 2);
      send(1, 'h80, 1, 64'h0123456789abcdef);
      send(1, 'hc0, 1, 64'h8877665544332211);
      wait (scoreboard[1].checked == submitted[1]);
      if (!saw_large_snoop)
        `uvm_fatal("SNP_SLOT", "No snoop used a Home slot beyond RN bank-ID range")
      hold_memory = 0;
      settle();
      send(0, 'h80);
      settle();  // Dirty owner at RN2 supplies the newest data and downgrades.
      send(0, 'h80, 1, 64'hfedcba9876543210);
      settle();
      send(0, 'h88, 1, 64'hdecafbad76543210, 8'h3c);
      settle();
      send(1, 'h80);
      settle();
      send(0, 'h80, 1, 64'h445566778899aabb);
      settle();
      send(0, 'h280);
      settle();  // Same private index: CopyBackWriteData into shared L2.
      send(1, 'h480);
      settle();
      send(1, 'h680);
      settle();  // L2 same-set replacement flushes dirty data.
      send(1, 'h80);
      settle();
      send(0, ERROR_ADDRESS);
      settle();
      if(!saw_dbid_mismatch||!saw_reorder||!dirty_snoops||!copybacks||!memory_writes||errors!=1||acks!=full_lines)
        `uvm_fatal("SCENARIOS", $sformatf(
                   "dbid=%0b reorder=%0b dirty=%0d copybacks=%0d writes=%0d errors=%0d ack/lines=%0d/%0d",
                   saw_dbid_mismatch,
                   saw_reorder,
                   dirty_snoops,
                   copybacks,
                   memory_writes,
                   errors,
                   acks,
                   full_lines
                   ))
      foreach (ack_pending[id])
        if (ack_pending[id]) `uvm_fatal("PENDING_ACK", "Unmatched completion acknowledgement")
      for (int core = 0; core < 2; core++)
        if (commands[core].size()) `uvm_fatal("PENDING_CPU", "Unmatched accepted CPU command")
      `uvm_info(
          "CHECKS",
          $sformatf(
              "canonical 2RN-F x2banks + shared L2: CPU checked=%0d/%0d full64B=%0d snoopLines=%0d dirtySnoops=%0d copybacks=%0d memoryWrites=%0d peakMSHR=%0d NDERR=%0d",
              scoreboard[0].checked, scoreboard[1].checked, full_lines, snoop_lines, dirty_snoops,
              copybacks, memory_writes, peak, errors), UVM_LOW)
    endtask
    function void final_phase(uvm_phase phase);
      coherence_ref_destroy(architectural);
      coherence_ref_destroy(backing);
    endfunction
  endclass
endpackage
