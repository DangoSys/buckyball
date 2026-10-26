module boot_tb;
  import uvm_pkg::*;
  import boot_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  boot_if bus (clock);
  BootRom dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual boot_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
