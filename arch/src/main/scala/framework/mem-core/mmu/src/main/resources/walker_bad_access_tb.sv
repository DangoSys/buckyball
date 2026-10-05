`include "walker_config.svh"
module walker_bad_access_tb;
  import uvm_pkg::*;
  import walker_negative_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  walker_control_if control (clock);
  stream_if #(`WALK_REQ_WIDTH) source_if (
      clock,
      control.reset
  );
  stream_if #(`WALK_RESP_WIDTH) sink_if (
      clock,
      control.reset
  );
  stream_if #(`WALK_ACCESS_WIDTH) access_if (
      clock,
      control.reset
  );
  stream_if #(`WALK_RESULT_WIDTH) result_if (
      clock,
      control.reset
  );
  Walker dut (
      .clock(clock),
      .reset(control.reset),
      .io_config_mode(control.mode),
      .io_config_rootPpn(control.root_ppn),
      `include "walker_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual walker_control_if)::set(null, "uvm_test_top", "control", control);
    uvm_config_db#(virtual stream_if #(`WALK_REQ_WIDTH))::set(null, "uvm_test_top", "req",
                                                              source_if);
    uvm_config_db#(virtual stream_if #(`WALK_RESP_WIDTH))::set(null, "uvm_test_top", "resp",
                                                               sink_if);
    uvm_config_db#(virtual stream_if #(`WALK_ACCESS_WIDTH))::set(null, "uvm_test_top", "access",
                                                                 access_if);
    uvm_config_db#(virtual stream_if #(`WALK_RESULT_WIDTH))::set(null, "uvm_test_top", "result",
                                                                 result_if);
    uvm_config_db#(int)::set(null, "uvm_test_top", "invalid_case", 2);
    run_test("protocol_test");
  end
endmodule
