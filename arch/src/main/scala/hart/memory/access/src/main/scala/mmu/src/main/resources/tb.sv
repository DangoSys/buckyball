module walker_tb;
  import uvm_pkg::*;
  import mmu_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  mmu_if bus (clock);
  assign bus.reset = timing.reset;
  Walker dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual mmu_if)::set(null, "uvm_test_top*", "vif", bus);
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top*", "timing", timing);
    run_test();
  end
endmodule
