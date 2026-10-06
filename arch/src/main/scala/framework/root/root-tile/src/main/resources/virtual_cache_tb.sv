`include "virtual_cache_config.svh"
`ifndef VIRTUAL_CACHE_TB
`define VIRTUAL_CACHE_TB virtual_cache_tb
`endif
module `VIRTUAL_CACHE_TB;
  import uvm_pkg::*;
  import virtual_cache_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  virtual_cache_if control (clock);
  stream_if #(`VM_AUTH_WIDTH) auth_req (
      clock,
      control.reset
  );
  stream_if #(1) auth_resp (
      clock,
      control.reset
  );
  stream_if #(`VM_REQ_WIDTH) source_if (
      clock,
      control.reset
  );
  stream_if #(`VM_RESP_WIDTH) sink_if (
      clock,
      control.reset
  );
  stream_if #(`VM_MEMREQ_WIDTH) mem_req (
      clock,
      control.reset
  );
  stream_if #(`VM_MEMRESP_WIDTH) mem_resp (
      clock,
      control.reset
  );
  stream_if #(`VM_UNCACHED_WIDTH) uncached_req (
      clock,
      control.reset
  );
  stream_if #(`VM_URESP_WIDTH) uncached_resp (
      clock,
      control.reset
  );
  VirtualCacheSystem dut (
      .clock(clock),
      .reset(control.reset),
      .io_active(control.active),
      .io_outstanding(control.outstanding),
      `include "virtual_cache_ports.svh"
  );
  // Observe existing HN boundary; no DUT logic is added. rxDat/rxRsp ready
  // are structurally true and optimized out of emitted RTL.
  always @(posedge clock) begin
    if (control.reset) begin
      control.conflict_wb = 0;
      control.conflict_dbid = 0;
      control.conflict_beats = 0;
      control.conflict_ack = 0;
    end else if (control.conflict_trace) begin
      if (dut.caches.home.home.io_req_valid && dut.caches.home.home.io_req_ready) begin
        if (dut.caches.home.home.io_req_bits_opcode == 7'h1b) control.conflict_wb++;
        $display("[CONFLICT_CHI] REQ op=%h txn=%h addr=%h", dut.caches.home.home.io_req_bits_opcode,
                 dut.caches.home.home.io_req_bits_txnId, dut.caches.home.home.io_req_bits_addr);
      end
      if (dut.caches.home.home.io_rsp_valid && dut.caches.home.home.io_rsp_ready) begin
        if (dut.caches.home.home.io_rsp_bits_opcode == 5'h05) control.conflict_dbid++;
        $display("[CONFLICT_CHI] RSP op=%h txn=%h dbid=%h", dut.caches.home.home.io_rsp_bits_opcode,
                 dut.caches.home.home.io_rsp_bits_txnId, dut.caches.home.home.io_rsp_bits_dbid);
      end
      if (dut.caches.home.home.io_rxDat_valid && dut.caches.home.home.io_rxDat_bits_opcode==4'h2) begin
        control.conflict_beats++;
        $display(
            "[CONFLICT_CHI] RXDAT op=%h txn=%h beat=%h", dut.caches.home.home.io_rxDat_bits_opcode,
            dut.caches.home.home.io_rxDat_bits_txnId, dut.caches.home.home.io_rxDat_bits_dataId);
      end
      if (dut.caches.home.home.io_rxRsp_valid && dut.caches.home.home.io_rxRsp_bits_opcode==5'h02) begin
        control.conflict_ack++;
        $display("[CONFLICT_CHI] RXRSP op=%h txn=%h", dut.caches.home.home.io_rxRsp_bits_opcode,
                 dut.caches.home.home.io_rxRsp_bits_txnId);
      end
    end
  end
  initial begin
    int invalid_case;
    uvm_config_db#(virtual stream_if #(`VM_AUTH_WIDTH))::set(null, "uvm_test_top", "auth_req",
                                                             auth_req);
    uvm_config_db#(virtual stream_if #(1))::set(null, "uvm_test_top", "auth_resp", auth_resp);
    uvm_config_db#(virtual virtual_cache_if)::set(null, "uvm_test_top", "control", control);
    uvm_config_db#(virtual stream_if #(`VM_REQ_WIDTH))::set(null, "uvm_test_top", "req", source_if);
    uvm_config_db#(virtual stream_if #(`VM_RESP_WIDTH))::set(null, "uvm_test_top", "resp", sink_if);
    uvm_config_db#(virtual stream_if #(`VM_MEMREQ_WIDTH))::set(null, "uvm_test_top", "mem_req",
                                                               mem_req);
    uvm_config_db#(virtual stream_if #(`VM_MEMRESP_WIDTH))::set(null, "uvm_test_top", "mem_resp",
                                                                mem_resp);
    uvm_config_db#(virtual stream_if #(`VM_UNCACHED_WIDTH))::set(null, "uvm_test_top",
                                                                 "uncached_req", uncached_req);
    uvm_config_db#(virtual stream_if #(`VM_URESP_WIDTH))::set(null, "uvm_test_top", "uncached_resp",
                                                              uncached_resp);
    if ($value$plusargs("invalid_case=%d", invalid_case))
      uvm_config_db#(int)::set(null, "uvm_test_top", "invalid_case", invalid_case);
    run_test("protocol_test");
  end
endmodule
