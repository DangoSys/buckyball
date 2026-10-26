module tbus_router_tb;
  import uvm_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  tbus_if bus (clock);
  Router dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual tbus_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test();
  end
endmodule
