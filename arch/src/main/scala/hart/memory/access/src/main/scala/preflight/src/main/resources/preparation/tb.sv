module preparation_tb;
  import uvm_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  preparation_if bus (clock);
  Preparation dut (
      .clock(clock),
      .reset(bus.reset),
      `include "preparation_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual preparation_if)::set(null, "uvm_test_top.env.*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
