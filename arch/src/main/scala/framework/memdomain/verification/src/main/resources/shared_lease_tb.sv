module shared_lease_tb;
  import uvm_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  ip_control_if ctl (clock);
  `include "shared_lease_signals.svh"
SharedMemBackend dut (
      `include "shared_lease_ports.svh"
  );
  `include "shared_lease_clocking.svh"
  int checks = 0, invalid_case = 0;
  task automatic check(bit good, string reason);
    checks++;
    if (!good) `uvm_fatal("SHARED_LEASE", reason)
  endtask
  task automatic configure(bit alloc, int bank, int group_id = 0, bit transfer = 0, int source = 0);
    @(negedge clock);
    io_config_bits_alloc = alloc;
    io_config_bits_vbank_id = bank;
    io_config_bits_group_id = group_id;
    io_config_bits_transfer = transfer;
    io_config_bits_source_bank_id = source;
    io_config_valid = 1;
    do @(sample); while (!sample.io_config_ready);
    @(negedge clock);
    io_config_valid = 0;
  endtask
  task automatic request(bit release_lease, int bank = 8, int group_id = 0);
    @(negedge clock);
    io_lease_request_bits_release = release_lease;
    io_lease_request_bits_vbank = bank;
    io_lease_request_bits_group = group_id;
    io_lease_request_valid = 1;
    do @(sample); while (!sample.io_lease_request_ready);
    @(negedge clock);
    io_lease_request_valid = 0;
  endtask
  task automatic reply(longint unsigned expected);
    repeat (5) begin
      @(sample);
      check(sample.io_lease_response_valid && sample.io_lease_response_bits == expected,
            "lease response must remain stable under backpressure");
      check(!sample.io_lease_request_ready, "accepted second lease before response");
    end
    @(negedge clock);
    io_lease_response_ready = 1;
    @(sample);
    check(sample.io_lease_response_valid, "response lost");
    @(negedge clock);
    io_lease_response_ready = 0;
  endtask
  task automatic data_access(bit write, bit [127:0] value);
    @(negedge clock);
    io_physical_0_request_valid = 1;
    io_physical_0_request_bits_address = 128;
    io_physical_0_request_bits_write = write;
    io_physical_0_request_bits_data = value;
    io_physical_0_request_bits_mask = '1;
    do @(sample); while (!sample.io_physical_0_request_ready);
    @(negedge clock);
    io_physical_0_request_valid = 0;
    do @(sample); while (!sample.io_physical_0_response_valid);
    check(!sample.io_physical_0_response_bits_error, "leased bank data access failed");
    if (!write)
      check(sample.io_physical_0_response_bits_data == value, "leased bank data mismatch");
    @(negedge clock);
    io_physical_0_response_ready = 1;
    @(sample);
    @(negedge clock);
    io_physical_0_response_ready = 0;
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 100us);
    run_test("protocol_test");
  end
  initial begin
    `include "shared_lease_init.svh"
    void'($value$plusargs("invalid_case=%d", invalid_case));
    wait (ctl.start);
    repeat (4) @(sample);
    @(negedge clock);
    reset = 0;
    io_config_bits_hart_id = 64'h100000002;
    io_config_bits_is_shared = 1;
    io_leaseOwnerHartId = 64'h100000002;
    configure(1, 8);
    configure(1, 8, 1);
    request(0, 8, 1);
    reply(128);
    if (invalid_case == 1) configure(0, 8);
    if (invalid_case == 2) configure(1, 8);
    if (invalid_case == 3) configure(0, 9, 0, 1, 8);
    if (invalid_case == 4) begin
      request(1, 8, 0);
    end
    if (invalid_case == 5) begin
      request(0, 10, 0);
    end
    if (invalid_case) begin
      repeat (8) @(sample);
      `uvm_fatal("SHARED_LEASE", "expected RTL assertion absent")
    end
    request(0, 8, 1);
    reply(128);
    request(1, 8, 1);
    reply(0);
    data_access(1, 128'hffeeddccbbaa99887766554433221100);
    data_access(0, 128'hffeeddccbbaa99887766554433221100);
    request(1, 8, 1);
    reply(0);
    // Config wins when both are offered; lease then observes the new mapping.
    @(negedge clock);
    io_config_valid = 1;
    io_config_bits_alloc = 1;
    io_config_bits_vbank_id = 9;
    io_config_bits_group_id = 0;
    io_lease_request_valid = 1;
    io_lease_request_bits_release = 0;
    io_lease_request_bits_vbank = 9;
    io_lease_request_bits_group = 0;
    @(sample);
    check(sample.io_config_ready && !sample.io_lease_request_ready, "config priority deadlock");
    @(negedge clock);
    io_config_valid = 0;
    @(sample);
    check(sample.io_lease_request_ready, "lease not accepted after config");
    @(negedge clock);
    io_lease_request_valid = 0;
    reply(256);
    request(1, 9);
    reply(0);
    configure(0, 9, 0, 1, 8);
    request(0, 9, 2);
    reply(128);
    request(1, 9, 2);
    reply(0);
    configure(0, 9);
    configure(1, 8);
    request(0);  // Reset discards an unconsumed response and its lease.
    @(negedge clock);
    reset = 1;
    repeat (3) @(sample);
    @(negedge clock);
    reset = 0;
    repeat (3) begin
      @(sample);
      check(!sample.io_lease_response_valid, "old response survived reset");
    end
    configure(1, 8);
    request(0);
    reply(0);
    request(1);
    reply(0);
    configure(0, 8);
    `uvm_info("SHARED_LEASE", $sformatf(
              "%0d checks: leases, data sharing, config priority, backpressure, reset", checks),
              UVM_LOW)
    ctl.done = 1;
  end
endmodule
