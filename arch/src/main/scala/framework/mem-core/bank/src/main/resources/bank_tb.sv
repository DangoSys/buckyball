module bank_tb;
  import uvm_pkg::*;
  import bank_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  bank_if source_if (clock);
  bank_if sink_if (clock);
  Bank dut (
      .clock(clock),
      .reset(source_if.reset),
      .io_request_valid(source_if.valid),
      .io_request_ready(source_if.ready),
      .io_request_bits_addr(source_if.addr),
      .io_request_bits_write(source_if.write),
      .io_request_bits_data(source_if.data),
      .io_request_bits_mask(source_if.mask),
      .io_request_bits_tag(source_if.tag),
      .io_response_valid(sink_if.valid),
      .io_response_ready(sink_if.ready),
      .io_response_bits_data(sink_if.data),
      .io_response_bits_tag(sink_if.tag),
      .io_response_bits_error(sink_if.error)
  );
  initial begin
    uvm_config_db#(virtual bank_if)::set(null, "uvm_test_top*", "request", source_if);
    uvm_config_db#(virtual bank_if)::set(null, "uvm_test_top*", "response", sink_if);
    run_test("protocol_test");
  end
endmodule
