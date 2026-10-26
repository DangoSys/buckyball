module rob_tb;
  import uvm_pkg::*;
  import rob_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  rob_if bus (clock);
  Rob dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual rob_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test();
  end
endmodule
