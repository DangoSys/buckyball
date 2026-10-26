module scoreboard_tb;
  import uvm_pkg::*;
  import scoreboard_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  bank_if bus (clock);
  BankScoreboard dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual bank_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test();
  end
endmodule
