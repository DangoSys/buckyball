module cache_tb;
  import uvm_pkg::*;
  import cache_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  cache_request_if source_if (clock);
  cache_response_if sink_if (clock);
  assign sink_if.reset = source_if.reset;
  Cache dut (
      .clock(clock),
      .reset(source_if.reset),
      .io_request_valid(source_if.valid),
      .io_request_ready(source_if.ready),
      .io_request_bits_id(source_if.id),
      .io_request_bits_op(source_if.op),
      .io_request_bits_addr(source_if.addr),
      .io_request_bits_way(source_if.way),
      .io_request_bits_data(source_if.data),
      .io_request_bits_mask(source_if.mask),
      .io_request_bits_metadata(source_if.metadata),
      .io_request_bits_eligible(source_if.eligible),
      .io_response_valid(sink_if.valid),
      .io_response_ready(sink_if.ready),
      .io_response_bits_id(sink_if.id),
      .io_response_bits_hit(sink_if.hit),
      .io_response_bits_available(sink_if.available),
      .io_response_bits_entryValid(sink_if.entry_valid),
      .io_response_bits_way(sink_if.way),
      .io_response_bits_addr(sink_if.addr),
      .io_response_bits_data(sink_if.data),
      .io_response_bits_metadata(sink_if.metadata)
  );
  initial begin
    uvm_config_db#(virtual cache_request_if)::set(null, "uvm_test_top*", "request", source_if);
    uvm_config_db#(virtual cache_response_if)::set(null, "uvm_test_top*", "response", sink_if);
    run_test("protocol_test");
  end
endmodule
