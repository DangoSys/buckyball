class home_range_test extends base_test;
  `uvm_component_utils(home_range_test)
  virtual coherence_control_if control;
  virtual stream_if #(`COH_REQ_WIDTH) req;
  virtual stream_if #(`COH_TXRSP_WIDTH) rsp;
  virtual stream_if #(`COH_RSP_WIDTH) rx_rsp;
  virtual stream_if #(`COH_DAT_WIDTH) rx_dat;
  virtual stream_if #(`COH_TXDAT_WIDTH) dat;
  virtual stream_if #(`COH_SNP_CHANNEL_WIDTH) snp;
  virtual stream_if #(`COH_MEMREQ_WIDTH) mem_req;
  virtual stream_if #(`COH_MEMRESP_WIDTH) mem_resp;
  bit expected[int];
  int sent, received;
  covergroup ranges with function sample (int node, int lines);
    option.per_instance = 1;
    nodes: coverpoint node {bins both[] = {[1 : 2]};}
    lengths: coverpoint lines {bins single = {1}; bins half = {32}; bins full = {64};}
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    ranges = new();
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual coherence_control_if)::get(
            this, "", "control", control
        ) || !uvm_config_db#(virtual stream_if #(`COH_REQ_WIDTH))::get(
            this, "", "req", req
        ) || !uvm_config_db#(virtual stream_if #(`COH_TXRSP_WIDTH))::get(
            this, "", "tx_rsp", rsp
        ) || !uvm_config_db#(virtual stream_if #(`COH_RSP_WIDTH))::get(
            this, "", "rx_rsp", rx_rsp
        ) || !uvm_config_db#(virtual stream_if #(`COH_DAT_WIDTH))::get(
            this, "", "rx_dat", rx_dat
        ) || !uvm_config_db#(virtual stream_if #(`COH_TXDAT_WIDTH))::get(
            this, "", "tx_dat", dat
        ) || !uvm_config_db#(virtual stream_if #(`COH_SNP_CHANNEL_WIDTH))::get(
            this, "", "snp", snp
        ) || !uvm_config_db#(virtual stream_if #(`COH_MEMREQ_WIDTH))::get(
            this, "", "mem_req", mem_req
        ) || !uvm_config_db#(virtual stream_if #(`COH_MEMRESP_WIDTH))::get(
            this, "", "mem_resp", mem_resp
        ))
      `uvm_fatal("VIF", "Home range interfaces missing")
  endfunction
  task send_range(int node, longint unsigned first, int last, int txn);
    req_t packet = '0;
    int   key = (node << 12) | txn;
    evidence::check(!expected.exists(key), "Range transaction reused before completion");
    expected[key] = 1;
    packet[`CF(REQ, SRCID)] = node;
    packet[`CF(REQ, TGTID)] = `COH_HOME;
    packet[`CF(REQ, TXNID)] = txn;
    packet[`CF(REQ, OPCODE)] = node == 1 ? `COH_CLEAN_SHARED : `COH_CLEAN_INVALID;
    packet[`CF(REQ, ADDR)] = first;
    packet[`CF(REQ, SIZE)] = 6;
    packet[`CF(REQ, MEMATTR)] = 12;
    packet[`CF(REQ, SNPATTR)] = 1;
    packet[`CF(REQ, ALLOWRETRY)] = 1;
    packet[`CF(REQ, MAINTENANCERANGE)] = 1;
    packet[`CF(REQ, MAINTENANCELASTLINE)] = last;
    @(negedge control.clock);
    req.valid = 1;
    req.bits  = packet;
    do @(posedge control.clock); while (!req.ready);
    @(negedge control.clock);
    req.valid = 0;
    sent++;
    ranges.sample(node, last - ((first >> 6) & 63) + 1);
  endtask
  task execute();
    req.valid = 0;
    req.bits = 0;
    rx_rsp.valid = 0;
    rx_rsp.bits = 0;
    rx_dat.valid = 0;
    rx_dat.bits = 0;
    mem_resp.valid = 0;
    mem_resp.bits = 0;
    rsp.ready = 0;
    dat.ready = 1;
    snp.ready = 1;
    mem_req.ready = 1;
    control.reset = 1;
    repeat (4) @(negedge control.clock);
    control.reset = 0;
    fork
      begin
        int cycle = 0;
        forever begin
          @(negedge control.clock);
          rsp.ready = (cycle++ % 7) < 3;
        end
      end
      begin
        forever begin
          @(posedge control.clock);
          evidence::check(!mem_req.valid && !snp.valid && !dat.valid,
                          "Cold range must not issue data traffic or CPU snoops");
          if (rsp.valid && rsp.ready) begin
            int node = rsp.bits[`CF(TXRSP, TGTID)];
            int key = (node << 12) | rsp.bits[`CF(TXRSP, TXNID)];
            evidence::check(expected.exists(key), "Duplicate or unknown range completion");
            evidence::check(rsp.bits[`CF(TXRSP, SRCID)] == `COH_HOME && rsp.bits[`CF(TXRSP, OPCODE)
                            ] == `COH_COMP && rsp.bits[`CF(TXRSP, RESPERR)] == 0,
                            "Invalid range completion");
            expected.delete(key);
            received++;
          end
        end
      end
    join_none
    send_range(1, 'h1000, 63, 7);
    send_range(2, 'h2800, 63, 9);
    wait (received == 2);
    send_range(1, 'h3fc0, 63, 7);
    send_range(2, 'h4000, 63, 9);
    wait (received == 4);
    evidence::check(sent == received && expected.num() == 0, "Range completion count mismatch");
    repeat (20) @(negedge control.clock);
    disable fork;
  endtask
endclass
