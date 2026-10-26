module tss_tb;
  import uvm_pkg::*;
  import tss_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  tss_if bus (clock);
  Store dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual tss_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
