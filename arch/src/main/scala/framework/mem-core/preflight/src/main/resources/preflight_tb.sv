`ifdef PREFLIGHT_LINE_READ
`include "preflight64_config.svh"
`define PREFLIGHT_TOP preflight64_tb
`else
`include "preflight_config.svh"
`define PREFLIGHT_TOP preflight_tb
`endif
module `PREFLIGHT_TOP;
  import uvm_pkg::*;
  import preflight_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  preflight_control_if ctl (clock);
  stream_if #(`PF_CMD_WIDTH) cmd (
      clock,
      ctl.reset
  );
  stream_if #(`PF_OUT_WIDTH) prepared (
      clock,
      ctl.reset
  );
  stream_if #(`PF_AUTH_WIDTH) authorization (
      clock,
      ctl.reset
  );
  stream_if #(`PF_PERMIT_WIDTH) permission (
      clock,
      ctl.reset
  );
  stream_if #(`PF_PTE_WIDTH) pte_req (
      clock,
      ctl.reset
  );
  stream_if #(`PF_PTERESP_WIDTH) pte_resp (
      clock,
      ctl.reset
  );
  Preflight dut (
      .clock(clock),
      .reset(ctl.reset),
      `include "preflight_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual preflight_control_if)::set(null, "uvm_test_top", "ctl", ctl);
    uvm_config_db#(virtual stream_if #(`PF_CMD_WIDTH))::set(null, "uvm_test_top", "cmd", cmd);
    uvm_config_db#(virtual stream_if #(`PF_OUT_WIDTH))::set(null, "uvm_test_top", "prepared",
                                                            prepared);
    uvm_config_db#(virtual stream_if #(`PF_AUTH_WIDTH))::set(null, "uvm_test_top", "authorization",
                                                             authorization);
    uvm_config_db#(virtual stream_if #(`PF_PERMIT_WIDTH))::set(null, "uvm_test_top", "permission",
                                                               permission);
    uvm_config_db#(virtual stream_if #(`PF_PTE_WIDTH))::set(null, "uvm_test_top", "pte_req",
                                                            pte_req);
    uvm_config_db#(virtual stream_if #(`PF_PTERESP_WIDTH))::set(null, "uvm_test_top", "pte_resp",
                                                                pte_resp);
    run_test("protocol_test");
  end
endmodule
`undef PREFLIGHT_TOP
