module write_dma_tb;
  import uvm_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  dma_if bus (clock);
  WriteDma dut (
      `include "write_dma_binding.svh"
  );
  initial begin
    uvm_config_db#(virtual dma_if)::set(null, "uvm_test_top.env", "vif", bus);
    run_test("protocol_test");
  end
endmodule
