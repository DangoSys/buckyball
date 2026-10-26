module prm_tb;
  import uvm_pkg::*;
  import prm_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  bus_if bus (clock);
  PRM dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  if (!`PRM_DIFF_TEST) begin
    for (genvar b = 0; b < `PRM_BANKS; b++) assign bus.ports[b].hash = 0;
  end
  initial begin
    uvm_config_db#(virtual bus_if)::set(null, "uvm_test_top.env.*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
