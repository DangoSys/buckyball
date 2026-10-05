class env extends uvm_env;
  `uvm_component_utils(env)
  virtual coherence_control_if control;
  virtual stream_if #(`COH_REQ_WIDTH) req;
  virtual stream_if #(`COH_RSP_WIDTH) rx_rsp;
  virtual stream_if #(`COH_DAT_WIDTH) rx_dat;
  virtual stream_if #(`COH_TXRSP_WIDTH) tx_rsp;
  virtual stream_if #(`COH_TXDAT_WIDTH) tx_dat;
  virtual stream_if #(`COH_SNP_CHANNEL_WIDTH) snp;
  virtual stream_if #(`COH_MEMREQ_WIDTH) mem_req;
  virtual stream_if #(`COH_MEMRESP_WIDTH) mem_resp;
  keyed_scoreboard #(completion) scoreboard;
  chandle model;
  req_t requests[int];
  logic [511:0] assembled[int];
  bit [`COH_BEATS-1:0] received[int];
  int data_dbid[int];
  logic [2:0] data_permission[int];
  logic [1:0] data_error[int];
  int expected_order[$];
  bit expected_error[int];
  private_line_t cpus[`COH_AGENTS][longint unsigned];
  bit [511:0] memory[longint unsigned];
  bit dirty_addresses[longint unsigned];
  int clean_shared_requests[longint unsigned];
  rsp_t rsp_queue[$], ack_queue[$];
  dat_t dat_queue[$];
  mem_t mem_queue[$];
  bit [`COH_MSHRS-1:0] hold_ack = 0;
  bit vary_idle_payload = 0;
  bit idle_payload_one = 0;
  bit hold_memory = 0, reverse_memory = 0, saw_reorder = 0;
  bit rsp_active = 0, dat_active = 0, mem_active = 0;
  rsp_t rsp_packet;
  dat_t dat_packet;
  mem_t mem_packet;
  int memory_writes = 0, memory_reads = 0, peak = 0;
  int cycle = 0;
  int snoops = 0;
  bit hold_rsp = 0, hold_snp = 0, hold_mem_req = 0;
  bit hold_copyback = 0;
  bit fault_enabled = 0, write_fault_enabled = 0;
  longint unsigned fault_addr;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    scoreboard = keyed_scoreboard#(completion)::type_id::create("scoreboard", this);
    model = coherence_ref_create();
    if (!uvm_config_db#(virtual coherence_control_if)::get(
            this, "", "control", control
        ) || !uvm_config_db#(virtual stream_if #(`COH_REQ_WIDTH))::get(
            this, "", "req", req
        ) || !uvm_config_db#(virtual stream_if #(`COH_RSP_WIDTH))::get(
            this, "", "rx_rsp", rx_rsp
        ) || !uvm_config_db#(virtual stream_if #(`COH_DAT_WIDTH))::get(
            this, "", "rx_dat", rx_dat
        ) || !uvm_config_db#(virtual stream_if #(`COH_TXRSP_WIDTH))::get(
            this, "", "tx_rsp", tx_rsp
        ) || !uvm_config_db#(virtual stream_if #(`COH_TXDAT_WIDTH))::get(
            this, "", "tx_dat", tx_dat
        ) || !uvm_config_db#(virtual stream_if #(`COH_SNP_CHANNEL_WIDTH))::get(
            this, "", "snp", snp
        ) || !uvm_config_db#(virtual stream_if #(`COH_MEMREQ_WIDTH))::get(
            this, "", "mem_req", mem_req
        ) || !uvm_config_db#(virtual stream_if #(`COH_MEMRESP_WIDTH))::get(
            this, "", "mem_resp", mem_resp
        ))
      `uvm_fatal("VIF", "Coherence interfaces missing")
  endfunction
  function void cpu_write(int node, longint unsigned addr, bit [511:0] value);
    if (!cpus[node-1].exists(addr) || !cpus[node-1][addr].valid || !cpus[node-1][addr].unique_owner)
      `uvm_fatal("CPU", "write without unique permission")
    cpus[node-1][addr].data  = value;
    cpus[node-1][addr].dirty = 1;
    coherence_ref_write(model, addr, value);
    dirty_addresses[addr] = 1;
  endfunction
  function void finish(completion item);
    int key = (item.node << 12) | item.txn;
    int position = -1;
    item.set_transaction_id(key);
    foreach (expected_order[i]) if (expected_order[i] == key) position = i;
    if (position < 0) `uvm_fatal("ORDER", "unexpected completion")
    if (position != 0) saw_reorder = 1;
    expected_order.delete(position);
    scoreboard.actual_export.write(item);
    if (requests[key][`CF(REQ, OPCODE)] == `COH_CLEAN_SHARED) begin
      longint unsigned address = requests[key][`CF(REQ, ADDR)];
      clean_shared_requests[address]--;
      if (clean_shared_requests[address] == 0) clean_shared_requests.delete(address);
    end
    requests.delete(key);
  endfunction

  task run_phase(uvm_phase phase);
    forever begin
      @(posedge control.clock);
      if (control.reset) begin
        if (dirty_addresses.num() != 0)
          `uvm_fatal("RESET", "test reset before dirty data reached memory")
        rx_rsp.valid <= 0;
        rx_rsp.bits <= '0;
        rx_dat.valid <= 0;
        rx_dat.bits <= '0;
        mem_resp.valid <= 0;
        mem_resp.bits <= '0;
        tx_rsp.ready <= 0;
        tx_dat.ready <= 0;
        snp.ready <= 0;
        mem_req.ready <= 0;
        rsp_active = 0;
        dat_active = 0;
        mem_active = 0;
        rsp_queue.delete();
        ack_queue.delete();
        dat_queue.delete();
        mem_queue.delete();
        foreach (cpus[i]) cpus[i].delete();
      end else begin
        cycle++;
        tx_rsp.ready <= !hold_rsp && (cycle % 8) >= 4;
        tx_dat.ready <= (cycle % 8) >= 4;
        snp.ready <= !hold_snp && (cycle % 8) >= 4;
        mem_req.ready <= !hold_mem_req && (cycle % 8) >= 4;
        if (control.outstanding > peak) peak = control.outstanding;
        if (rsp_active && rx_rsp.ready) rsp_active = 0;
        if (dat_active && rx_dat.ready) dat_active = 0;
        if (mem_active && mem_resp.ready) mem_active = 0;
        if (req.valid && req.ready) begin
          completion item = completion::type_id::create("expected");
          int node = req.bits[`CF(REQ, SRCID)];
          int txn = req.bits[`CF(REQ, TXNID)];
          int key = (node << 12) | txn;
          int opcode = req.bits[`CF(REQ, OPCODE)];
          longint unsigned addr = req.bits[`CF(REQ, ADDR)];
          bit [511:0] value;
          requests[key] = req.bits;
          if (opcode == `COH_CLEAN_SHARED) begin
            if (!clean_shared_requests.exists(addr)) clean_shared_requests[addr] = 0;
            clean_shared_requests[addr]++;
          end
          expected_order.push_back(key);
          item.node = node;
          item.txn = txn;
          item.is_data=opcode==`COH_READ_SHARED || opcode==`COH_READ_NSD || opcode==`COH_READ_UNIQUE;
          item.opcode=item.is_data ? `COH_COMP_DATA : (opcode==`COH_WRITEBACK ? `COH_COMP_DBID : `COH_COMP);
          item.error = expected_error.exists(key) && expected_error[key] ? 3 : 0;
          item.permission=item.is_data && item.error==0 ? (opcode==`COH_READ_UNIQUE ? 2 : 1) : 0;
          if (item.is_data && !item.error) begin
            coherence_ref_read(model, addr, value);
            item.data = value;
          end
          item.set_transaction_id(key);
          scoreboard.expected_export.write(item);
          if (opcode == `COH_EVICT && cpus[node-1].exists(addr)) begin
            if (cpus[node-1][addr].valid && cpus[node-1][addr].dirty)
              `uvm_fatal("CPU", "dirty line sent as Evict")
            cpus[node-1][addr].valid = 0;
          end
        end
        if (snp.valid && snp.ready) begin
          int node = snp.bits[`CF(SNP, TARGET)];
          int txn = snp.bits[`CF(SNP, TXNID)];
          int opcode = snp.bits[`CF(SNP, OPCODE)];
          longint unsigned addr = longint'(snp.bits[`CF(SNP, ADDR)]) << 3;
          private_line_t line;
          bit invalidate = opcode == `COH_SNP_UNIQUE || opcode == `COH_SNP_INVALID || opcode == `COH_SNP_MAKE_INVALID;
          bit discard = opcode == `COH_SNP_MAKE_INVALID;
          if (node < 1 || node > `COH_AGENTS || snp.bits[`CF(SNP, SRCID)] != `COH_HOME)
            `uvm_fatal("SNOOP", "invalid destination/source")
          snoops++;
          line = cpus[node-1].exists(addr) ? cpus[node-1][addr] : '0;
          if ((opcode == `COH_SNP_INVALID || opcode == `COH_SNP_CLEAN_SHARED || discard) && snp.bits[
              `CF(SNP, RETTOSRC)
              ] !== 0)
            `uvm_fatal("SNOOP", "Maintenance snoop must clear RetToSrc")
          if (line.valid && !discard && (line.dirty || (!line.unique_owner && snp.bits[
              `CF(SNP, RETTOSRC)
              ]))) begin
            for (int b = 0; b < `COH_BEATS; b++) begin
              dat_t packet = '0;
              packet[`CF(DAT, TGTID)] = `COH_HOME;
              packet[`CF(DAT, SRCID)] = node;
              packet[`CF(DAT, TXNID)] = txn;
              packet[`CF(DAT, OPCODE)] = `COH_SNP_DATA;
              packet[`CF(DAT, RESP)] = (invalidate ? 0 : 1) | (line.dirty ? 4 : 0);
              packet[`CF(DAT, DATAID)] = b * (`COH_DATA_BITS / 128);
              packet[`CF(DAT, BE)] = '1;
              packet[`CF(DAT, DATA)] = line.data[b*`COH_DATA_BITS+:`COH_DATA_BITS];
              dat_queue.push_back(packet);
            end
          end else begin
            rsp_t packet = '0;
            packet[`CF(RSP, TGTID)]  = `COH_HOME;
            packet[`CF(RSP, SRCID)]  = node;
            packet[`CF(RSP, TXNID)]  = txn;
            packet[`CF(RSP, OPCODE)] = `COH_SNP_RESP;
            packet[`CF(RSP, RESP)]   = line.valid && !invalidate ? 1 : 0;
            rsp_queue.push_back(packet);
          end
          if (line.valid) begin
            line.valid = !invalidate;
            line.unique_owner = 0;
            line.dirty = 0;
            cpus[node-1][addr] = line;
          end
        end
        if (tx_rsp.valid && tx_rsp.ready) begin
          completion item = completion::type_id::create("rsp");
          int node = tx_rsp.bits[`CF(TXRSP, TGTID)];
          int txn = tx_rsp.bits[`CF(TXRSP, TXNID)];
          int key = (node << 12) | txn;
          int dbid = tx_rsp.bits[`CF(TXRSP, DBID)];
          if (!requests.exists(
                  key
              ) || tx_rsp.bits[
              `CF(TXRSP, SRCID)
              ] != `COH_HOME || dbid >= `COH_MSHRS)
            `uvm_fatal("RSP", "unexpected response routing")
          item.node = node;
          item.txn = txn;
          item.opcode = tx_rsp.bits[`CF(TXRSP, OPCODE)];
          item.error = tx_rsp.bits[`CF(TXRSP, RESPERR)];
          item.permission = tx_rsp.bits[`CF(TXRSP, RESP)];
          if (item.opcode == `COH_COMP_DBID) begin
            longint unsigned addr = requests[key][`CF(REQ, ADDR)];
            private_line_t   line = cpus[node-1].exists(addr) ? cpus[node-1][addr] : '0;
            for (int b = 0; b < `COH_BEATS; b++) begin
              dat_t packet = '0;
              packet[`CF(DAT, TGTID)] = `COH_HOME;
              packet[`CF(DAT, SRCID)] = node;
              packet[`CF(DAT, TXNID)] = dbid;
              packet[`CF(DAT, OPCODE)] = `COH_COPYBACK_DATA;
              packet[
              `CF(DAT, RESP)
              ] = line.valid ? (line.unique_owner ? 2 : 1) | (line.dirty ? 4 : 0) : 0;
              packet[`CF(DAT, DATAID)] = b * (`COH_DATA_BITS / 128);
              packet[`CF(DAT, BE)] = line.valid ? '1 : '0;
              packet[`CF(DAT, DATA)] = line.data[b*`COH_DATA_BITS+:`COH_DATA_BITS];
              dat_queue.push_back(packet);
            end
            if (line.valid) begin
              line.valid = 0;
              cpus[node-1][addr] = line;
            end
          end
          finish(item);
        end
        if (tx_dat.valid && tx_dat.ready) begin
          int node = tx_dat.bits[`CF(TXDAT, TGTID)];
          int txn = tx_dat.bits[`CF(TXDAT, TXNID)];
          int key = (node << 12) | txn;
          int b = tx_dat.bits[`CF(TXDAT, DATAID)] / (`COH_DATA_BITS / 128);
          int dbid = tx_dat.bits[`CF(TXDAT, DBID)];
          if (!requests.exists(
                  key
              ) || tx_dat.bits[
              `CF(TXDAT, SRCID)
              ] != `COH_HOME || tx_dat.bits[
              `CF(TXDAT, HOMENID)
              ] != `COH_HOME || dbid >= `COH_MSHRS || b >= `COH_BEATS)
            `uvm_fatal("DATA", "unexpected response routing")
          if (!received.exists(key)) begin
            received[key] = '0;
            assembled[key] = '0;
            data_dbid[key] = dbid;
            data_permission[key] = tx_dat.bits[`CF(TXDAT, RESP)];
            data_error[key] = tx_dat.bits[`CF(TXDAT, RESPERR)];
          end
          if (tx_dat.bits[
              `CF(TXDAT, OPCODE)
              ] !== `COH_COMP_DATA || tx_dat.bits[
              `CF(TXDAT, DATAID)
              ] % (`COH_DATA_BITS / 128) != 0 || tx_dat.bits[
              `CF(TXDAT, RESP)
              ] !== data_permission[key] || tx_dat.bits[
              `CF(TXDAT, RESPERR)
              ] !== data_error[key] || tx_dat.bits[
              `CF(TXDAT, BE)
              ] !== (data_error[key] != 0 ? `COH_TXDAT_BE_WIDTH'(0) : {`COH_TXDAT_BE_WIDTH{1'b1}}))
            `uvm_fatal("DATA", "inconsistent completion data header")
          if (received[key][b] || dbid != data_dbid[key])
            `uvm_fatal("DATA", "duplicate beat or inconsistent DBID")
          assembled[key][b*`COH_DATA_BITS+:`COH_DATA_BITS] = tx_dat.bits[`CF(TXDAT, DATA)];
          received[key][b] = 1;
          if (&received[key]) begin
            completion item = completion::type_id::create("data");
            rsp_t ack = '0;
            longint unsigned addr = requests[key][`CF(REQ, ADDR)];
            item.node = node;
            item.txn = txn;
            item.is_data = 1;
            item.opcode = tx_dat.bits[`CF(TXDAT, OPCODE)];
            item.error = tx_dat.bits[`CF(TXDAT, RESPERR)];
            item.permission = tx_dat.bits[`CF(TXDAT, RESP)];
            item.data = assembled[key];
            if (!item.error)
              cpus[node-1][addr] = '{
                  valid: 1,
                  unique_owner: (item.permission == 2),
                  dirty: 0,
                  data: item.data
              };
            ack[`CF(RSP, TGTID)]  = `COH_HOME;
            ack[`CF(RSP, SRCID)]  = node;
            ack[`CF(RSP, TXNID)]  = dbid;
            ack[`CF(RSP, OPCODE)] = `COH_COMP_ACK;
            ack_queue.push_back(ack);
            finish(item);
            received.delete(key);
            assembled.delete(key);
            data_dbid.delete(key);
            data_permission.delete(key);
            data_error.delete(key);
          end
        end
        if (mem_req.valid && mem_req.ready) begin
          mem_t packet = '0;
          longint unsigned addr = mem_req.bits[`CF(MEMREQ, ADDR)];
          bit [511:0] value;
          packet[`CF(MEMRESP, ID)] = mem_req.bits[`CF(MEMREQ, ID)];
          if (mem_req.bits[`CF(MEMREQ, WRITE)]) begin
            coherence_ref_read(model, addr, value);
            if (mem_req.bits[`CF(MEMREQ, DATA)] !== value || mem_req.bits[`CF(MEMREQ, MASK)] !== '1)
              `uvm_fatal("MEMORY", "dirty victim writeback lost data")
            foreach (cpus[i])
            if (cpus[i].exists(
                    addr
                ) && cpus[i][addr].valid && (!clean_shared_requests.exists(
                    addr
                ) || cpus[i][addr].dirty || cpus[i][addr].unique_owner ||
                    cpus[i][addr].data !== value))
              `uvm_fatal(
                  "INCLUSION",
                  "DDR clean completed while a private copy was dirty, unique or inconsistent")
            if (write_fault_enabled && addr == fault_addr) begin
              packet[`CF(MEMRESP, ERROR)] = 1;
            end else begin
              memory[addr] = value;
              dirty_addresses.delete(addr);
            end
            memory_writes++;
          end else begin
            if (memory.exists(addr)) value = memory[addr];
            else coherence_ref_initial(addr, value);
            packet[`CF(MEMRESP, DATA)]  = value;
            packet[`CF(MEMRESP, ERROR)] = fault_enabled && addr == fault_addr;
            memory_reads++;
          end
          mem_queue.push_back(packet);
        end
        if (!rsp_active) begin
          if (rsp_queue.size() != 0) begin
            rsp_packet = rsp_queue.pop_front();
            rsp_active = 1;
          end else begin
            int chosen = -1;
            foreach (ack_queue[j])
            if (chosen < 0 && !hold_ack[ack_queue[j][`CF(RSP, TXNID)]]) chosen = j;
            if (chosen >= 0) begin
              rsp_packet = ack_queue[chosen];
              ack_queue.delete(chosen);
              rsp_active = 1;
            end
          end
        end
        if (!dat_active && dat_queue.size() != 0 && (!hold_copyback || dat_queue[0][
            `CF(DAT, OPCODE)
            ] != `COH_COPYBACK_DATA)) begin
          dat_packet = dat_queue.pop_front();
          dat_active = 1;
        end
        if (!mem_active && !hold_memory && mem_queue.size() != 0) begin
          mem_packet = reverse_memory ? mem_queue.pop_back() : mem_queue.pop_front();
          mem_active = 1;
        end
        rx_rsp.valid <= rsp_active;
        rx_rsp.bits <= rsp_active || !vary_idle_payload ? rsp_packet : {`COH_RSP_WIDTH{idle_payload_one}};
        rx_dat.valid <= dat_active;
        rx_dat.bits <= dat_active || !vary_idle_payload ? dat_packet : {`COH_DAT_WIDTH{idle_payload_one}};
        mem_resp.valid <= mem_active;
        mem_resp.bits <= mem_active || !vary_idle_payload ? mem_packet : {`COH_MEMRESP_WIDTH{idle_payload_one}};
      end
    end
  endtask
  function void final_phase(uvm_phase phase);
    coherence_ref_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
