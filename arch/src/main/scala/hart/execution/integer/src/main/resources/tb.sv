module integer_tb;
  import uvm_pkg::*;
  import integer_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  port_if bus (
      clock,
      timing.reset
  );
  Integer dut (
      .clock(clock),
      .reset(bus.reset),
      .io_in1(bus.a),
      .io_in2(bus.rhs),
      .io_fn(bus.fn),
      .io_dw(bus.dw),
      .io_out(bus.value),
      .io_adder_out(bus.adder),
      .io_cmp_out(bus.comparison)
  );
  initial begin
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top*", "timing", timing);
    uvm_config_db#(virtual port_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
