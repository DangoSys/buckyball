`include "profile.svh"
`ifndef DDR_TB
`define DDR_TB bridge_tb
`endif
module `DDR_TB;
  import uvm_pkg::*;
  import ddr_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  ddr_control_if ctl (clock);
  stream_if #(`DDR_REQ_WIDTH)
      req0 (
          clock,
          ctl.reset
      ),
      req1 (
          clock,
          ctl.reset
      );
  stream_if #(`DDR_RESP_WIDTH)
      resp0 (
          clock,
          ctl.reset
      ),
      resp1 (
          clock,
          ctl.reset
      );
  stream_if #(`DDR_A_WIDTH)
      aw (
          clock,
          ctl.reset
      ),
      ar (
          clock,
          ctl.reset
      );
  stream_if #(`DDR_W_WIDTH) w (
      clock,
      ctl.reset
  );
  stream_if #(`DDR_B_WIDTH) b (
      clock,
      ctl.reset
  );
  stream_if #(`DDR_R_WIDTH) r (
      clock,
      ctl.reset
  );
  Bridge dut (
      .clock(clock),
      .reset(ctl.reset),
      `include `DDR_PORTS
  );
  initial begin
    int invalid_case;
    uvm_config_db#(virtual ddr_control_if)::set(null, "uvm_test_top", "ctl", ctl);
    uvm_config_db#(virtual stream_if #(`DDR_REQ_WIDTH))::set(null, "uvm_test_top", "req0", req0);
    uvm_config_db#(virtual stream_if #(`DDR_REQ_WIDTH))::set(null, "uvm_test_top", "req1", req1);
    uvm_config_db#(virtual stream_if #(`DDR_RESP_WIDTH))::set(null, "uvm_test_top", "resp0", resp0);
    uvm_config_db#(virtual stream_if #(`DDR_RESP_WIDTH))::set(null, "uvm_test_top", "resp1", resp1);
    uvm_config_db#(virtual stream_if #(`DDR_A_WIDTH))::set(null, "uvm_test_top", "aw", aw);
    uvm_config_db#(virtual stream_if #(`DDR_A_WIDTH))::set(null, "uvm_test_top", "ar", ar);
    uvm_config_db#(virtual stream_if #(`DDR_W_WIDTH))::set(null, "uvm_test_top", "w", w);
    uvm_config_db#(virtual stream_if #(`DDR_B_WIDTH))::set(null, "uvm_test_top", "b", b);
    uvm_config_db#(virtual stream_if #(`DDR_R_WIDTH))::set(null, "uvm_test_top", "r", r);
    if ($value$plusargs("invalid_case=%d", invalid_case))
      uvm_config_db#(int)::set(null, "uvm_test_top", "invalid_case", invalid_case);
    run_test("protocol_test");
  end
endmodule
