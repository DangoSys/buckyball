`include "config.svh"
module tls_tb;
  import uvm_pkg::*;
  import ip_pkg::*;
  import ip_control_test_pkg::*;
  import spm_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0;
  always #5 clock = ~clock;
  logic [1:0] reset = '1, active = '0;
  ip_control_if control (clock);
  virtual spm_if ports[4];
  chandle models[2];
  for (genvar i = 0; i < 4; i++) begin : endpoints
    spm_if bus (
        clock,
        reset[i/2]
    );
    initial ports[i] = bus;
  end
  for (genvar tile = 0; tile < 2; tile++) begin : tiles
    Store dut (
        .clock(clock),
        .reset(reset[tile]),
        .io_running(active[tile]),
        .io_cancel(endpoints[tile*2+1].bus.cancel),
        .io_busy(),
        .io_host_request_valid(endpoints[tile*2+0].bus.valid),
        .io_host_request_ready(endpoints[tile*2+0].bus.ready),
        .io_host_request_bits_address(endpoints[tile*2+0].bus.address),
        .io_host_request_bits_size(endpoints[tile*2+0].bus.size),
        .io_host_request_bits_write(endpoints[tile*2+0].bus.write),
        .io_host_request_bits_data(endpoints[tile*2+0].bus.data),
        .io_host_request_bits_mask(endpoints[tile*2+0].bus.mask),
        .io_host_response_valid(endpoints[tile*2+0].bus.response_valid),
        .io_host_response_ready(endpoints[tile*2+0].bus.response_ready),
        .io_host_response_bits_data(endpoints[tile*2+0].bus.result),
        .io_host_response_bits_error(endpoints[tile*2+0].bus.error),
        .io_local_request_valid(endpoints[tile*2+1].bus.valid),
        .io_local_request_ready(endpoints[tile*2+1].bus.ready),
        .io_local_request_bits_address(endpoints[tile*2+1].bus.address),
        .io_local_request_bits_size(endpoints[tile*2+1].bus.size),
        .io_local_request_bits_write(endpoints[tile*2+1].bus.write),
        .io_local_request_bits_data(endpoints[tile*2+1].bus.data),
        .io_local_request_bits_mask(endpoints[tile*2+1].bus.mask),
        .io_local_response_valid(endpoints[tile*2+1].bus.response_valid),
        .io_local_response_ready(endpoints[tile*2+1].bus.response_ready),
        .io_local_response_bits_data(endpoints[tile*2+1].bus.result),
        .io_local_response_bits_error(endpoints[tile*2+1].bus.error)
    );
  end
  initial begin
    foreach (models[tile]) models[tile] = spm_create(`LOCAL_BASE, `LOCAL_BYTES);
    #0;
    foreach (ports[i]) begin
      ports[i].model = models[i/2];
      ports[i].scoreboard = new($sformatf("port_%0d", i), null);
      ports[i].enabled = 1;
    end
    uvm_config_db#(virtual ip_control_if)::set(null, "uvm_test_top", "vif", control);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 100us);
    run_test("protocol_test");
  end
  final foreach (models[i]) spm_destroy(models[i]);
  initial begin : exercise
    logic [127:0] value;
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    for (int row = 0; row < `LOCAL_BYTES; row += 16) begin
      ports[0].access(`LOCAL_BASE + row, 4, 1, 0, '1);
      ports[2].access(`LOCAL_BASE + row, 4, 1, '1, '1);
    end
    @(negedge clock);
    active = 3;
    ports[1].access(`LOCAL_BASE, 3, 1, 64'h123456789abcdef0, 16'hff);
    ports[3].access(`LOCAL_BASE, 4, 0);
    ports[1].access(`LOCAL_BASE + 15, 0, 1, 8'ha5, 1);
    ports[1].access(`LOCAL_BASE, 4, 0);
    ports[1].send(`LOCAL_BASE, 4, 0);
    wait (ports[1].response_valid);
    repeat (5) @(negedge clock);
    ports[3].access(`LOCAL_BASE + 16, 4, 0);
    ports[1].take(value);
    ports[1].access(`LOCAL_BASE + 3, 2, 0);
    ports[1].access(`LOCAL_BASE + `LOCAL_BYTES, 3, 0);
    ports[1].access(64'h80000000, 3, 0);
    ports[1].access(`LOCAL_BASE, 5, 0);
    ports[1].access(`LOCAL_BASE + 8, 3, 1, 64'h1122334455667788, 16'ha5);
    ports[1].access(`LOCAL_BASE + 8, 3, 0);
    ports[1].access((64'h1 << 32) | `LOCAL_BASE, 3, 0);
    ports[1].access(64'hfffffffffffffff8, 3, 0);
    ports[1].access(`LOCAL_BASE, 0, 1, '1, 3);
    ports[1].send(`LOCAL_BASE, 3, 0);
    ports[1].cancel = 1;
    @(negedge clock);
    ports[1].cancel = 0;
    repeat (5) begin
      @(negedge clock);
      if (ports[1].response_valid) `uvm_fatal("CANCEL", "old TLS response survived cancellation")
    end
    ports[1].access(`LOCAL_BASE, 3, 0);
    ports[1].send(`LOCAL_BASE + 16, 0, 1, 8'h5a, 1);
    wait (ports[1].response_valid);
    @(negedge clock);
    reset[0]  = 1;
    active[0] = 0;
    repeat (2) @(negedge clock);
    reset[0] = 0;
    ports[0].access(`LOCAL_BASE + 16, 0, 0);
    ports[3].access(`LOCAL_BASE + 16, 0, 0);
    repeat (2) @(negedge clock);
    foreach (ports[i]) begin
      if (ports[i].scoreboard.expected_queue.size() || ports[i].scoreboard.actual_queue.size())
        `uvm_fatal("DRAIN", "local storage still has unmatched accesses")
      `uvm_info("CHECKED", $sformatf("port %0d: %0d comparisons", i, ports[i].scoreboard.checked),
                UVM_LOW)
    end
    control.done = 1;
  end
endmodule
