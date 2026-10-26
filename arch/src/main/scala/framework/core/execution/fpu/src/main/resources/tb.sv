module fpu_tb;
  import uvm_pkg::*;
  import fpu_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  port_if bus (
      clock,
      timing.reset
  );
  FPU dut (
      .clock(clock),
      .reset(bus.reset),
      .io_request_valid(bus.request_valid),
      .io_request_ready(bus.request_ready),
      .io_request_bits_instruction(bus.instruction),
      .io_request_bits_integer(bus.integer_value),
      .io_request_bits_a(bus.a),
      .io_request_bits_rhs(bus.rhs),
      .io_request_bits_c(bus.c),
      .io_request_bits_rounding(bus.rounding),
      .io_response_valid(bus.response_valid),
      .io_response_ready(bus.response_ready),
      .io_response_bits_data(bus.value),
      .io_response_bits_flags(bus.flags),
      .io_response_bits_integer(bus.integer_result),
      .io_response_bits_illegal(bus.illegal)
  );
  initial begin
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top*", "timing", timing);
    uvm_config_db#(virtual port_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
