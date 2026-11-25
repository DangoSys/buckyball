module spm_tb;
  import uvm_pkg::*;
  import spm_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  port_if bus (
      clock,
      timing.reset
  );
  Memory dut (
      .clock(clock),
      .reset(bus.reset),
      .io_port_request_valid(bus.request_valid),
      .io_port_request_ready(bus.request_ready),
      .io_port_request_bits_address(bus.address),
      .io_port_request_bits_size(bus.size),
      .io_port_request_bits_write(bus.write),
      .io_port_request_bits_data(bus.data),
      .io_port_request_bits_mask(bus.mask),
      .io_readOnly(bus.read_only),
      .io_busy(bus.busy),
      .io_port_response_valid(bus.response_valid),
      .io_port_response_ready(bus.response_ready),
      .io_port_response_bits_data(bus.value),
      .io_port_response_bits_error(bus.error)
  );
  initial begin
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top*", "timing", timing);
    uvm_config_db#(virtual port_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
