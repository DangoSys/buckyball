`include "preflight_config.svh"
module prepared_map_tb;
  import uvm_pkg::*;
  import prepared_map_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  prepared_map_control_if ctl (clock);
  stream_if #(`PF_RESERVE_WIDTH) reserve (
      clock,
      ctl.reset
  );
  stream_if #(`PF_OUT_WIDTH) prepared (
      clock,
      ctl.reset
  );
  stream_if #(`PF_MAP_READY_WIDTH) mapped (
      clock,
      ctl.reset
  );
  stream_if #(`PF_RELEASE_WIDTH) retire (
      clock,
      ctl.reset
  );
  assign mapped.ready = !ctl.reset && !ctl.hold_ready;
  PreparedMap dut (
      .clock(clock),
      .reset(ctl.reset),
      `include "prepared_map_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual prepared_map_control_if)::set(null, "uvm_test_top", "ctl", ctl);
    uvm_config_db#(virtual stream_if #(`PF_RESERVE_WIDTH))::set(null, "uvm_test_top", "reserve",
                                                                reserve);
    uvm_config_db#(virtual stream_if #(`PF_OUT_WIDTH))::set(null, "uvm_test_top", "prepared",
                                                            prepared);
    uvm_config_db#(virtual stream_if #(`PF_MAP_READY_WIDTH))::set(null, "uvm_test_top", "mapped",
                                                                  mapped);
    uvm_config_db#(virtual stream_if #(`PF_RELEASE_WIDTH))::set(null, "uvm_test_top", "retire",
                                                                retire);
    run_test("protocol_test");
  end
endmodule
