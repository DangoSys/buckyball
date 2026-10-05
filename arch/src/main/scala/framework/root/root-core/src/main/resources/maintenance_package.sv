`include "maintenance_config.svh"
package maintenance_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  `define MF(kind, field) `CORE_``kind``_``field``_OFFSET +: `CORE_``kind``_``field``_WIDTH
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual maintenance_control_if control;
    virtual stream_if #(`CORE_RANGE_WIDTH) request;
    virtual stream_if #(`CORE_ACK_WIDTH) response;
    virtual stream_if #(`CORE_REQ_WIDTH) req;
    virtual stream_if #(`CORE_RSP_WIDTH) rsp;
    int checked_lines = 0, checked_ranges = 0;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 50us;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual maintenance_control_if)::get(
              this, "", "control", control
          ) || !uvm_config_db#(virtual stream_if #(`CORE_RANGE_WIDTH))::get(
              this, "", "request", request
          ) || !uvm_config_db#(virtual stream_if #(`CORE_ACK_WIDTH))::get(
              this, "", "response", response
          ) || !uvm_config_db#(virtual stream_if #(`CORE_REQ_WIDTH))::get(
              this, "", "req", req
          ) || !uvm_config_db#(virtual stream_if #(`CORE_RSP_WIDTH))::get(
              this, "", "rsp", rsp
          ))
        `uvm_fatal("VIF", "Maintenance interfaces missing")
    endfunction
    task reset_all();
      control.cb.reset <= 1;
      control.cb.drained <= 0;
      request.valid <= 0;
      request.bits <= '0;
      response.ready <= 0;
      req.ready <= 0;
      rsp.valid <= 0;
      rsp.bits <= '0;
      repeat (3) @(control.cb);
      control.cb.reset <= 0;
      repeat (2) @(control.cb);
      if (!control.cb.idle || req.valid || response.valid)
        `uvm_fatal("RESET", "Joint reset retained work")
    endtask
    task start_range(int tag, int op, longint unsigned first, int lines);
      bit [`CORE_RANGE_WIDTH-1:0] bits = '0;
      bits[`MF(RANGE, TAG)] = tag;
      bits[`MF(RANGE, OP)] = op;
      bits[`MF(RANGE, FIRSTLINE)] = first;
      bits[`MF(RANGE, LASTLINE)] = first + (lines - 1) * 64;
      request.bits  <= bits;
      request.valid <= 1;
      do @(request.sample); while (!request.sample.ready);
      request.valid <= 0;
    endtask
    task line_completion(longint unsigned address, int opcode, bit error = 0);
      bit [`CORE_REQ_WIDTH-1:0] expected = '0;
      bit [`CORE_RSP_WIDTH-1:0] reply = '0;
      int home = `MAINTENANCE_HOME_BASE + ((address >> 6) % `MAINTENANCE_HOME_COUNT);
      expected[`MF(REQ, SRCID)] = `MAINTENANCE_NODE;
      expected[`MF(REQ, TGTID)] = home;
      expected[`MF(REQ, TXNID)] = `MAINTENANCE_TXN;
      expected[`MF(REQ, ADDR)] = address;
      expected[`MF(REQ, SIZE)] = 6;
      expected[`MF(REQ, OPCODE)] = opcode;
      expected[`MF(REQ, SNPATTR)] = 1;
      expected[`MF(REQ, MEMATTR)] = 12;
      expected[`MF(REQ, ALLOWRETRY)] = 1;
      do @(req.sample); while (!req.sample.valid);
      if (req.sample.bits !== expected)
        `uvm_fatal("REQ", $sformatf(
                   "CMO mismatch at %h: %h != %h", address, req.sample.bits, expected))
      repeat (3) begin
        @(req.sample);
        if (!req.sample.valid || req.sample.bits !== expected || response.valid)
          `uvm_fatal("STALL", "CMO changed or completed while request was stalled")
      end
      req.ready <= 1;
      @(req.sample);
      req.ready <= 0;
      repeat (5) begin
        @(rsp.sample);
        if (req.valid || response.valid)
          `uvm_fatal("EARLY", "Range advanced before actual Home completion")
      end
      reply[`MF(RSP, SRCID)]   = home;
      reply[`MF(RSP, TGTID)]   = `MAINTENANCE_NODE;
      reply[`MF(RSP, TXNID)]   = `MAINTENANCE_TXN;
      reply[`MF(RSP, OPCODE)]  = 4;
      reply[`MF(RSP, RESPERR)] = error ? 3 : 0;
      rsp.bits  <= reply;
      rsp.valid <= 1;
      do @(rsp.sample); while (!rsp.sample.ready);
      rsp.valid <= 0;
      checked_lines++;
    endtask
    task finish_range(int tag, bit ok);
      bit [`CORE_ACK_WIDTH-1:0] expected = '0;
      expected[`MF(ACK, TAG)] = tag;
      expected[`MF(ACK, OK)]  = ok;
      do @(response.sample); while (!response.sample.valid);
      repeat (4) begin
        if (response.sample.bits !== expected || req.valid || request.ready)
          `uvm_fatal("ACK", "Bad completion or accepted new range before retirement")
        @(response.sample);
      end
      response.ready <= 1;
      @(response.sample);
      response.ready <= 0;
      @(control.cb);
      if (!control.cb.idle) `uvm_fatal("IDLE", "Completed range did not retire")
      checked_ranges++;
    endtask
    task execute();
      reset_all();
`ifdef MAINTENANCE_BAD_RESPONSE
      rsp.valid <= 1;
      repeat (3) @(control.cb);
      `uvm_fatal("MISSING_ASSERT", "Unissued Home response was not rejected")
`else
      for (int op = 0; op < 3; op++) begin
        int lines = 3 - op;
        control.cb.drained <= 0;
        start_range(16 + op, op, 64'h80000000, lines);
        repeat (4) begin
          @(control.cb);
          if (req.valid || response.valid) `uvm_fatal("DRAIN", "CMO overtook older CPU accesses")
        end
        control.cb.drained <= 1;
        for (int i = 0; i < lines; i++) line_completion(64'h80000000 + i * 64, 8 + op);
        finish_range(16 + op, 1);
      end
      start_range(255, 1, 64'h80000100, 4);
      line_completion(64'h80000100, 9);
      line_completion(64'h80000140, 9, 1);
      finish_range(255, 0);
      start_range(0, 0, 64'hfffffffffc0, 1);
      line_completion(64'hfffffffffc0, 8);
      finish_range(0, 1);
      control.cb.drained <= 0;
      start_range(42, 0, 64'h80000000, 1);
      reset_all();
      control.cb.drained <= 1;
      start_range(43, 2, 64'h80000000, 1);
      do @(req.sample); while (!req.sample.valid);
      req.ready <= 1;
      @(req.sample);
      req.ready <= 0;
      reset_all();
      control.cb.drained <= 1;
      start_range(44, 2, 64'h80000040, 1);
      line_completion(64'h80000040, 10);
      finish_range(44, 1);
      if (checked_lines != 10 || checked_ranges != 6)
        `uvm_fatal("COUNTS", "Maintenance scenarios incomplete")
      `uvm_info("MAINTENANCE",
                "Checked 10 CMO lines, 6 ranges, delayed Home completion, errors and joint reset",
                UVM_LOW)
`endif
    endtask
  endclass
endpackage
