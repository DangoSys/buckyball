module storage_tb;
  import uvm_pkg::*;
  import storage_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  storage_if access[2] (clock);
  `define CLIENT(I) \
    .io_access_``I``_request_valid(access[I].valid), \
    .io_access_``I``_request_ready(access[I].ready), \
    .io_access_``I``_request_bits_address(access[I].address), \
    .io_access_``I``_request_bits_write(access[I].write), \
    .io_access_``I``_request_bits_data(access[I].data), \
    .io_access_``I``_request_bits_mask(access[I].mask), \
    .io_access_``I``_response_valid(access[I].response_valid), \
    .io_access_``I``_response_ready(access[I].response_ready), \
    .io_access_``I``_response_bits_data(access[I].result), \
    .io_access_``I``_response_bits_error(access[I].error)
  Storage dut (
      .clock  (clock),
      .reset  (access[0].reset),
      `CLIENT(0),
      `CLIENT(1),
      .io_busy()
  );
  `undef CLIENT
  initial begin
    uvm_config_db#(storage_vif)::set(null, "uvm_test_top", "access0", access[0]);
    uvm_config_db#(storage_vif)::set(null, "uvm_test_top", "access1", access[1]);
    run_test();
  end
endmodule
