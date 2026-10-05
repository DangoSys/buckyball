`include "tile_config.svh"
module tile_tb;
  import uvm_pkg::*;
  import tile_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  tile_if control (clock);
  stream_if #(`TILE_MEMREQ_WIDTH) mem_req (
      clock,
      control.reset
  );
  stream_if #(`TILE_MEMRESP_WIDTH) mem_resp (
      clock,
      control.reset
  );
  stream_if #(`TILE_UNCACHED_WIDTH)
      uncached_req_0 (
          clock,
          control.reset
      ),
      uncached_req_1 (
          clock,
          control.reset
      );
  stream_if #(`TILE_URESP_WIDTH)
      uncached_resp_0 (
          clock,
          control.reset
      ),
      uncached_resp_1 (
          clock,
          control.reset
      );
  Tile dut (
      .clock(clock),
      .reset(control.reset),
      .io_resetVector(64'h80000000),
      .io_time(64'b0),
      .io_outstanding(control.outstanding),
      `include "tile_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual tile_if)::set(null, "uvm_test_top", "control", control);
    uvm_config_db#(virtual stream_if #(`TILE_MEMREQ_WIDTH))::set(null, "uvm_test_top", "mem_req",
                                                                 mem_req);
    uvm_config_db#(virtual stream_if #(`TILE_MEMRESP_WIDTH))::set(null, "uvm_test_top", "mem_resp",
                                                                  mem_resp);
    uvm_config_db#(virtual stream_if #(`TILE_UNCACHED_WIDTH))::set(
        null, "uvm_test_top", "uncached_req_0", uncached_req_0);
    uvm_config_db#(virtual stream_if #(`TILE_UNCACHED_WIDTH))::set(
        null, "uvm_test_top", "uncached_req_1", uncached_req_1);
    uvm_config_db#(virtual stream_if #(`TILE_URESP_WIDTH))::set(null, "uvm_test_top",
                                                                "uncached_resp_0", uncached_resp_0);
    uvm_config_db#(virtual stream_if #(`TILE_URESP_WIDTH))::set(null, "uvm_test_top",
                                                                "uncached_resp_1", uncached_resp_1);
    run_test("protocol_test");
  end
endmodule
