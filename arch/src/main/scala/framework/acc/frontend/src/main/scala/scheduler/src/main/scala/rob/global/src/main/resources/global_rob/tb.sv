module global_rob_tb;
  import uvm_pkg::*;
  import global_rob_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  global_if bus (clock);
  assign bus.reset = timing.reset;
  GlobalROB dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual global_if)::set(null, "uvm_test_top*", "vif", bus);
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top.env.reset", "timing", timing);
    run_test("protocol_test");
  end
endmodule
