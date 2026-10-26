module bat_tb;
  import uvm_pkg::*;
  import bat_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  bank_if bus (clock);
  BankAliasTable dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual bank_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test();
  end
endmodule
