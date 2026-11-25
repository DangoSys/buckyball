module muldiv_tb;
  import uvm_pkg::*;
  import muldiv_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  port_if bus (
      clock,
      timing.reset
  );
  MulDiv dut (
      .clock(clock),
      .reset(bus.reset),
      .io_request_valid(bus.request_valid),
      .io_request_ready(bus.request_ready),
      .io_request_bits_a(bus.a),
      .io_request_bits_rhs(bus.rhs),
      .io_request_bits_operation(bus.operation),
      .io_request_bits_word(bus.word),
      .io_response_valid(bus.response_valid),
      .io_response_ready(bus.response_ready),
      .io_response_bits(bus.value)
  );
  initial begin
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top*", "timing", timing);
    uvm_config_db#(virtual port_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
