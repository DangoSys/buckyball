`include "cpu_mem_config.svh"
`ifndef CPU_MEM_TOP
`define CPU_MEM_TOP cpu_mem_tb
`endif
module `CPU_MEM_TOP;
  import uvm_pkg::*;
  import cpu_mem_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  cpu_control_if control (clock);
  stream_if #(`CPU_REQ_WIDTH) request (
      clock,
      control.reset
  );
  stream_if #(`CPU_RESP_WIDTH) response (
      clock,
      control.reset
  );
  stream_if #(`CPU_CACHE_WIDTH) cache_request (
      clock,
      control.reset
  );
  stream_if #(`CPU_CRESULT_WIDTH) cache_response (
      clock,
      control.reset
  );
  stream_if #(`CPU_UNCACHED_WIDTH) uncached_request (
      clock,
      control.reset
  );
  stream_if #(`CPU_URESULT_WIDTH) uncached_response (
      clock,
      control.reset
  );
  int unsigned in_flight = 0;
  bit issued = 0, returned = 0, uncached = 0;
  always @(posedge clock) begin
    if (control.reset) begin
      in_flight = 0;
      issued = 0;
      returned = 0;
    end else begin
      if (in_flight != 0 && request.ready)
        $fatal(1, "CPU port became ready before consuming its response");
      if (request.valid && request.ready) begin
        if (in_flight != 0) $fatal(1, "CPU memory accepted overlapping requests");
        in_flight = 1;
        issued = 0;
        returned = 0;
      end
      if ((cache_request.valid && cache_request.ready) || (uncached_request.valid && uncached_request.ready)) begin
        if (in_flight != 1 || issued) $fatal(1, "Unexpected or duplicate backend request");
        issued   = 1;
        uncached = uncached_request.valid && uncached_request.ready;
      end
      if ((cache_response.valid && cache_response.ready) || (uncached_response.valid && uncached_response.ready)) begin
        if (in_flight != 1 || !issued || returned ||
          uncached != (uncached_response.valid && uncached_response.ready))
          $fatal(1, "Unexpected or duplicate backend response");
        returned = 1;
      end
      if (response.valid && response.ready) begin
        if (in_flight != 1 || (issued && !returned)) $fatal(1, "Unexpected CPU result");
        in_flight = 0;
      end
    end
  end
  CpuMem dut (
      .clock(clock),
      .reset(control.reset),
      `include "cpu_mem_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual cpu_control_if)::set(null, "uvm_test_top*", "control", control);
    uvm_config_db#(virtual stream_if #(`CPU_REQ_WIDTH))::set(null, "uvm_test_top*", "request",
                                                             request);
    uvm_config_db#(virtual stream_if #(`CPU_RESP_WIDTH))::set(null, "uvm_test_top*", "response",
                                                              response);
    uvm_config_db#(virtual stream_if #(`CPU_CACHE_WIDTH))::set(null, "uvm_test_top*",
                                                               "cache_request", cache_request);
    uvm_config_db#(virtual stream_if #(`CPU_CRESULT_WIDTH))::set(null, "uvm_test_top*",
                                                                 "cache_response", cache_response);
    uvm_config_db#(virtual stream_if #(`CPU_UNCACHED_WIDTH))::set(
        null, "uvm_test_top*", "uncached_request", uncached_request);
    uvm_config_db#(virtual stream_if #(`CPU_URESULT_WIDTH))::set(
        null, "uvm_test_top*", "uncached_response", uncached_response);
    run_test("protocol_test");
  end
endmodule
