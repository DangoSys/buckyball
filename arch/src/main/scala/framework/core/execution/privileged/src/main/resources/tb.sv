module privileged_tb;
  import uvm_pkg::*;
  import privileged_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  privileged_if bus (clock);
  Privileged dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual privileged_if)::set(null, "uvm_test_top.env.*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
