module decoder_tb;
  import uvm_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  decoder_if bus (clock);
  Decoder dut (
      .clock(clock),
      .reset(bus.reset),
      `include "decoder_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual decoder_if)::set(null, "uvm_test_top.env.*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
