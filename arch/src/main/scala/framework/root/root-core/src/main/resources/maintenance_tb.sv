`include "maintenance_config.svh"
`ifndef MAINTENANCE_TB
`define MAINTENANCE_TB maintenance_tb
`endif
module `MAINTENANCE_TB;
  import uvm_pkg::*;
  import maintenance_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  maintenance_control_if control (clock);
  stream_if #(`CORE_RANGE_WIDTH) source_if (
      clock,
      control.reset
  );
  stream_if #(`CORE_ACK_WIDTH) sink_if (
      clock,
      control.reset
  );
  stream_if #(`CORE_REQ_WIDTH) chi_req (
      clock,
      control.reset
  );
  stream_if #(`CORE_RSP_WIDTH) chi_rsp (
      clock,
      control.reset
  );
  Maintenance dut (
      .clock(clock),
      .reset(control.reset),
      .io_drained(control.drained),
      .io_idle(control.idle),
      `include "maintenance_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual maintenance_control_if)::set(null, "uvm_test_top*", "control", control);
    uvm_config_db#(virtual stream_if #(`CORE_RANGE_WIDTH))::set(null, "uvm_test_top*", "request",
                                                                source_if);
    uvm_config_db#(virtual stream_if #(`CORE_ACK_WIDTH))::set(null, "uvm_test_top*", "response",
                                                              sink_if);
    uvm_config_db#(virtual stream_if #(`CORE_REQ_WIDTH))::set(null, "uvm_test_top*", "req",
                                                              chi_req);
    uvm_config_db#(virtual stream_if #(`CORE_RSP_WIDTH))::set(null, "uvm_test_top*", "rsp",
                                                              chi_rsp);
    run_test("protocol_test");
  end
endmodule
