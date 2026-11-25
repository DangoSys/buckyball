`include "config.svh"
module code_tb;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  import ant_store_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0;
  always #5 clock = ~clock;
  logic [1:0] reset = '1, active = '0;
  scenario_if control (clock);
  virtual spm_if ports[4];
  chandle models[2];
  for (genvar i = 0; i < 4; i++) begin : endpoints
    spm_if bus (
        clock,
        reset[i/2]
    );
    initial ports[i] = bus;
  end
  for (genvar hart = 0; hart < 2; hart++) begin : harts
    LocalStore dut (
        .clock(clock),
        .reset(reset[hart]),
        .io_running(active[hart]),
        .io_cancel(endpoints[hart*2+1].bus.cancel),
        .io_busy(),
        .io_host_request_valid(endpoints[hart*2+0].bus.valid),
        .io_host_request_ready(endpoints[hart*2+0].bus.ready),
        .io_host_request_bits_address(endpoints[hart*2+0].bus.address),
        .io_host_request_bits_size(endpoints[hart*2+0].bus.size),
        .io_host_request_bits_write(endpoints[hart*2+0].bus.write),
        .io_host_request_bits_data(endpoints[hart*2+0].bus.data),
        .io_host_request_bits_mask(endpoints[hart*2+0].bus.mask),
        .io_host_response_valid(endpoints[hart*2+0].bus.response_valid),
        .io_host_response_ready(endpoints[hart*2+0].bus.response_ready),
        .io_host_response_bits_data(endpoints[hart*2+0].bus.result),
        .io_host_response_bits_error(endpoints[hart*2+0].bus.error),
        .io_local_request_valid(endpoints[hart*2+1].bus.valid),
        .io_local_request_ready(endpoints[hart*2+1].bus.ready),
        .io_local_request_bits_address(endpoints[hart*2+1].bus.address),
        .io_local_request_bits_size(endpoints[hart*2+1].bus.size),
        .io_local_request_bits_write(endpoints[hart*2+1].bus.write),
        .io_local_request_bits_data(endpoints[hart*2+1].bus.data),
        .io_local_request_bits_mask(endpoints[hart*2+1].bus.mask),
        .io_local_response_valid(endpoints[hart*2+1].bus.response_valid),
        .io_local_response_ready(endpoints[hart*2+1].bus.response_ready),
        .io_local_response_bits_data(endpoints[hart*2+1].bus.result),
        .io_local_response_bits_error(endpoints[hart*2+1].bus.error)
    );
  end
  initial begin
    foreach (models[hart]) models[hart] = spm_create(0, `CODE_BYTES);
    #0;
    foreach (ports[i]) begin
      ports[i].model = models[i/2];
      ports[i].scoreboard = new($sformatf("port_%0d", i), null);
      ports[i].enabled = 1;
    end
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 100us);
    run_test("protocol_test");
  end
  final foreach (models[i]) spm_destroy(models[i]);
  initial begin : exercise
    logic [127:0] value;
    int unsigned half = `CODE_BYTES / 2;
    int unsigned addresses[4] = '{0, `CODE_BYTES / 2 - 16, `CODE_BYTES / 2, `CODE_BYTES - 16};
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    foreach (addresses[i]) begin
      ports[0].access(addresses[i], 4, 1, 128'h0123456789abcdef0000000000000000 + i, '1);
      ports[2].access(addresses[i], 4, 1, 128'hfedcba98765432100000000000000000 + i, '1);
    end
    active = 3;
    // The request address changes banks while its response is held.
    ports[1].send(half - 16, 4, 0);
    wait (ports[1].response_valid);
    ports[1].address = half;
    repeat (7) @(negedge clock);
    ports[3].access(half, 4, 0);
    ports[1].take(value);
    ports[1].send(half, 4, 0);
    wait (ports[1].response_valid);
    ports[1].address = 0;
    repeat (5) @(negedge clock);
    ports[1].take(value);
    foreach (addresses[i]) begin
      ports[1].access(addresses[i], 4, 0);
      ports[3].access(addresses[3-i], 4, 0);
    end
    // Accepted reads in either leaf drain and discard their response on cancellation.
    for (int bank = 0; bank < 2; bank++) begin
      ports[1].send(bank * half, 4, 0);
      ports[1].cancel = 1;
      @(negedge clock);
      ports[1].cancel = 0;
      repeat (5) begin
        @(negedge clock);
        evidence::check(!ports[1].response_valid, "cancelled code response drained");
      end
      active[0] = 0;
      ports[0].access(bank * half, 4, 1, 128'h7654321001234567 + bank, '1);
      active[0] = 1;
      ports[1].access(bank * half, 4, 0);
    end
    ports[1].access(`CODE_BYTES, 3, 0);
    ports[1].access(64'hfffffffffffffff8, 3, 0);
    ports[1].send(half, 4, 0);
    wait (ports[1].response_valid);
    @(negedge clock);
    reset[0]  = 1;
    active[0] = 0;
    repeat (2) @(negedge clock);
    reset[0] = 0;
    ports[0].access(half, 4, 0);
    foreach (ports[i]) begin
      evidence::check(ports[i].scoreboard.pending() == 0, "code port drained");
      `uvm_info("CHECKED", $sformatf("port %0d: %0d comparisons", i, ports[i].scoreboard.checked),
                UVM_LOW)
    end
    control.done = 1;
  end
endmodule
