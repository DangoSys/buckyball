`include "cache_system_config.svh"
module cache_system_tb;
  import uvm_pkg::*;
  import cache_system_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  cache_system_if control (clock);
  stream_if #(`COH_MEMREQ_WIDTH) mem_req (
      clock,
      control.reset
  );
  stream_if #(`COH_MEMRESP_WIDTH) mem_resp (
      clock,
      control.reset
  );
  CacheSystem dut (
      .clock(clock),
      .reset(control.reset),
      .io_active(control.active),
      .io_blockRequesterRsp(control.block_requester_rsp),
      .io_outstanding(control.outstanding),
      `include "cache_system_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual cache_system_if)::set(null, "uvm_test_top", "control", control);
    uvm_config_db#(virtual stream_if #(`COH_MEMREQ_WIDTH))::set(null, "uvm_test_top", "mem_req",
                                                                mem_req);
    uvm_config_db#(virtual stream_if #(`COH_MEMRESP_WIDTH))::set(null, "uvm_test_top", "mem_resp",
                                                                 mem_resp);
    run_test("protocol_test");
  end
endmodule
