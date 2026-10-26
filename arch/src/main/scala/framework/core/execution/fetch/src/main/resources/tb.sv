module fetch_tb;
  import uvm_pkg::*;
  import fetch_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  fetch_if bus (clock);
  assign bus.reset = timing.reset;
  Fetch dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual fetch_if)::set(null, "uvm_test_top*", "vif", bus);
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top*", "timing", timing);
    run_test();
  end
endmodule
