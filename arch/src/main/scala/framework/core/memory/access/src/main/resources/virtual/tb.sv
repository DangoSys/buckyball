module virtual_access_tb;
  import uvm_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  virtual_if bus (clock);
  VirtualAccess dut (
      .clock(clock),
      .reset(bus.reset),
      `include "virtual_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual virtual_if)::set(null, "uvm_test_top.env.*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
