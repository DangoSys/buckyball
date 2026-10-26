module tomem_tb;
  import uvm_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  tomem_if bus (clock);
  ToMem dut (
      .clock(clock),
      .reset(bus.reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual tomem_if)::set(null, "uvm_test_top.env.*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
