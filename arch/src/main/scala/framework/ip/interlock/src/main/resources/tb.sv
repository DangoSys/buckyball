module interlock_tb;
  import uvm_pkg::*;
  import interlock_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  interlock_if bus (clock);
  Interlock dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual interlock_if)::set(null, "uvm_test_top.*", "vif", bus);
    run_test();
  end
endmodule
