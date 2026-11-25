module interconnect_tb;
  import uvm_pkg::*;
  import interconnect_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  bus_if bus (
      clock,
      timing.reset
  );
  Interconnect dut (
      .clock(clock),
      .reset(timing.reset),
      .io_outstanding(bus.outstanding),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual bus_if)::set(null, "uvm_test_top.env.*", "vif", bus);
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top.env.reset", "timing", timing);
    run_test("protocol_test");
  end
endmodule
