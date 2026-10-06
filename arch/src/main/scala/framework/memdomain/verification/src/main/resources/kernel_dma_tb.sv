module kernel_dma_tb;
  import uvm_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  ip_control_if ctl (clock);
  `include "kernel_dma_signals.svh"
KernelDma dut (
      `include "kernel_dma_ports.svh"
  );
  `include "kernel_dma_clocking.svh"
stream_if #(64 + 32) source_if (
      clock,
      reset || io_kernel_abort
  );
  assign source_if.valid = io_request_valid;
  assign source_if.ready = io_request_ready;
  assign source_if.bits  = {io_request_bits_vaddr, io_request_bits_len};
  stream_if #(32) sink_if (
      clock,
      reset || io_kernel_abort
  );
  assign sink_if.valid = io_kernel_image_valid;
  assign sink_if.ready = io_kernel_image_ready;
  assign sink_if.bits  = io_kernel_image_bits;
  stream_if #(68) result_if (
      clock,
      reset
  );
  assign result_if.valid = io_kernel_result_valid;
  assign result_if.ready = io_kernel_result_ready;
  assign result_if.bits  = {io_kernel_result_bits_error, io_kernel_result_bits_address};
  int images = 0, results = 0, requests = 0, cases = 0, checked = 0;
  int expected_images = 0, expected_error = 0;
  int expected_rob = 0;
  longint unsigned base = 0, fault_address = 0;
  int length = 0;
  function automatic logic [31:0] word_value(int index);
    return 32'hc1700081 ^ (index * 32'h1020305);
  endfunction
  function automatic logic [127:0] beat_value(int first);
    logic [127:0] data;
    for (int i = 0; i < 4; i++) data[32*i+:32] = word_value(first + i);
    return data;
  endfunction
  task automatic verify_contract(bit condition, string message);
    if (!condition) `uvm_fatal("KERNEL_DMA", message)
  endtask
  task automatic observe();
    forever begin
      @(sample);
      if (!sample.reset) begin
        verify_contract(sample.io_footprint_valid == sample.io_kernel_busy,
                        "footprint lifetime differs from transaction ownership");
        if (sample.io_footprint_valid) begin
          longint unsigned rounded_span;
          rounded_span = (64'(length) + 15) & ~64'd15;
          verify_contract(
              sample.io_footprint_rob_id==expected_rob &&
            !sample.io_footprint_is_sub && sample.io_footprint_sub_rob_id==0 &&
            sample.io_footprint_baseVA==base && sample.io_footprint_rows==1 &&
            sample.io_footprint_columns==1 && sample.io_footprint_spanBytes==rounded_span &&
            sample.io_footprint_columnStride==0 && sample.io_footprint_rowStride==0 &&
            !sample.io_footprint_write,
              "captured footprint changed or lost full VA/rounded transport span");
          verify_contract(
              sample.io_footprint_fault_error==(expected_error==7 ? 7 : 0) &&
            sample.io_footprint_fault_address==(expected_error==7 ? base : 0),
              "footprint shape fault changed while stalled");
          checked++;
        end
        if (sample.io_request_valid && sample.io_request_ready) begin
          requests++;
          verify_contract(
              requests==1 && sample.io_request_bits_vaddr==base &&
            sample.io_request_bits_len==length && sample.io_request_bits_groups==1 &&
            sample.io_request_bits_stride==1 && !sample.io_request_bits_is_2d,
              "wrong DMA request or duplicate dispatch");
        end
        if (sample.io_kernel_image_valid && sample.io_kernel_image_ready) begin
          verify_contract(sample.io_kernel_busy && images < expected_images,
                          "bad/later image word escaped");
          verify_contract(sample.io_kernel_image_bits === word_value(images),
                          "image word order/value differs from independent sequence");
          images++;
          checked++;
        end
        if (sample.io_kernel_result_valid) begin
          verify_contract(sample.io_kernel_busy && !sample.io_kernel_load_ready,
                          "terminal result did not retain ownership");
          verify_contract(images == expected_images,
                          "terminal result before all permitted image words consumed");
          verify_contract(
              sample.io_kernel_result_bits_error==expected_error &&
            sample.io_kernel_result_bits_address==fault_address,
              "wrong terminal full status/address");
          if (sample.io_kernel_result_ready) results++;
          checked++;
        end
      end
    end
  endtask
  task automatic configure(longint unsigned address, int bytes, int words, int error = 0,
                           longint unsigned error_address = 0);
    @(negedge clock);
    verify_contract(!io_kernel_busy && io_kernel_load_ready,
                    "previous transaction did not release");
    base = address;
    length = bytes;
    expected_images = words;
    expected_error = error;
    fault_address = error_address;
    images = 0;
    results = 0;
    requests = 0;
    cases++;
    expected_rob = cases % (1 << $bits(io_rob_id));
    io_rob_id = expected_rob;
    io_kernel_image_ready = 0;
    io_kernel_result_ready = 0;
    io_request_ready = 0;
    io_response_valid = 0;
    io_response_bits_fault_error = 0;
    io_response_bits_fault_address = 0;
    io_kernel_load_bits_address = address;
    io_kernel_load_bits_bytes = bytes;
    io_kernel_load_valid = 1;
    do @(sample); while (!sample.io_kernel_load_ready);
    @(negedge clock);
    io_kernel_load_valid = 0;
    io_rob_id = ~expected_rob;
  endtask
  task automatic accept_request();
    wait (io_request_valid);
    repeat (4) begin
      @(sample);
      verify_contract(
          sample.io_kernel_busy && !sample.io_kernel_load_ready && !sample.io_kernel_result_valid,
          "request stalled ownership wrong");
    end
    @(negedge clock);
    io_request_ready = 1;
    do @(sample); while (!sample.io_request_valid);
    @(negedge clock);
    io_request_ready = 0;
  endtask
  task automatic response(int first, bit last, int error = 0, longint unsigned address = 0);
    @(negedge clock);
    io_response_bits_data = beat_value(first);
    io_response_bits_last = last;
    io_response_bits_addrcounter = first / 4;
    io_response_bits_fault_error = error;
    io_response_bits_fault_address = address;
    io_response_valid = 1;
    do @(sample); while (!sample.io_response_ready);
    @(negedge clock);
    io_response_valid = 0;
  endtask
  task automatic consume_words(int count);
    @(negedge clock);
    io_kernel_image_ready = 1;
    wait (images == count);
    @(negedge clock);
    io_kernel_image_ready = 0;
  endtask
  task automatic no_terminal(int cycles);
    repeat (cycles) begin
      @(sample);
      verify_contract(
          sample.io_kernel_busy && !sample.io_kernel_result_valid &&
        !sample.io_kernel_load_ready && results==0,
          "early terminal before accepted read drained");
    end
  endtask
  task automatic retire_result(int required_requests);
    wait (io_kernel_result_valid);
    repeat (8) @(sample);
    @(negedge clock);
    io_kernel_abort = 1;  // Held terminal must be immutable, including success.
    repeat (4) @(sample);
    @(negedge clock);
    io_kernel_abort = 0;
    io_kernel_result_ready = 1;
    do @(sample); while (!sample.io_kernel_result_valid);
    @(negedge clock);
    io_kernel_result_ready = 0;
    repeat (4) @(sample);
    @(negedge clock);
    verify_contract(
        results==1 && requests==required_requests && !io_kernel_busy && io_kernel_load_ready &&
      !io_kernel_result_valid && !io_kernel_image_valid && !io_footprint_valid,
        "duplicate result or ownership/footprint not released");
  endtask
  task automatic recovery();
    configure('h9000 + cases * 64, 4, 1);
    accept_request();
    response(0, 1);
    consume_words(1);
    retire_result(1);
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 100us);
    run_test("protocol_test");
  end
  initial begin
    `include "kernel_dma_init.svh"
    wait (ctl.start);
    repeat (4) @(sample);
    @(negedge clock);
    reset = 0;
    fork
      observe();
    join_none
    // A partial final beat emits exactly five words; success requires the final image handshake.
    configure('h1000, 20, 5);
    accept_request();
    response(0, 0);
    no_terminal(8);
    consume_words(4);
    response(4, 1);
    no_terminal(8);
    consume_words(5);
    retire_result(1);
    // Original unaligned VA is preserved; preflight later aligns [1,17) to [0,32).
    configure(1, 4, 1);
    accept_request();
    response(0, 1);
    no_terminal(8);
    consume_words(1);
    retire_result(1);
    // First and middle faults preserve all 64 address bits and drop later data while draining.
    configure('h2000, 48, 0, 3, 64'h123456789abcde00);
    accept_request();
    response(0, 0, 3, 64'h123456789abcde00);
    no_terminal(8);
    response(4, 0);
    no_terminal(8);
    response(8, 1, 4, 'h2222);
    retire_result(1);
    recovery();
    configure('h3000, 48, 4, 4, 64'hfedcba9876543210);
    accept_request();
    response(0, 0);
    consume_words(4);
    response(4, 0, 4, 64'hfedcba9876543210);
    no_terminal(8);
    response(8, 1);
    retire_result(1);
    recovery();
    // Abort an offered, unaccepted DMA request: cancellation, no request handshake.
    configure('h4000, 16, 0, 10, 'h4000);
    wait (io_request_valid);
    @(negedge clock);
    io_kernel_abort = 1;
    repeat (4) @(sample);
    @(negedge clock);
    io_kernel_abort = 0;
    retire_result(0);
    recovery();
    // Abort an accepted read: keep ownership until its terminal response is actually consumed.
    configure('h5000, 32, 0, 10, 'h5000);
    accept_request();
    @(negedge clock);
    io_kernel_abort = 1;
    no_terminal(8);
    @(negedge clock);
    io_kernel_abort = 0;
    response(0, 0);
    no_terminal(8);
    response(4, 1);
    retire_result(1);
    recovery();
    // Abort a buffered non-final output after one word: flush the remaining words and drain.
    configure('h6000, 32, 1, 10, 'h6000);
    accept_request();
    response(0, 0);
    consume_words(1);
    @(negedge clock);
    io_kernel_abort = 1;
    no_terminal(4);
    @(negedge clock);
    io_kernel_abort = 0;
    response(4, 1);
    retire_result(1);
    recovery();
    // Abort a fully buffered final beat with zero output acceptance.
    configure('h6100, 16, 0, 10, 'h6100);
    accept_request();
    response(0, 1);
    @(negedge clock);
    io_kernel_abort = 1;
    repeat (4) @(sample);
    @(negedge clock);
    io_kernel_abort = 0;
    retire_result(1);
    recovery();
    // An abort asserted before load acceptance is not a transaction and produces no result.
    @(negedge clock);
    io_kernel_abort = 1;
    io_kernel_load_valid = 1;
    repeat (8) begin
      @(sample);
      verify_contract(
          !sample.io_kernel_load_ready && !sample.io_kernel_busy &&
      !sample.io_kernel_result_valid && !sample.io_request_valid,
          "unaccepted cancelled load generated work");
    end
    @(negedge clock);
    io_kernel_load_valid = 0;
    io_kernel_abort = 0;
    recovery();
    // Shape errors never dispatch a DMA request.
    configure('h7000, 0, 0, 7, 'h7000);
    retire_result(0);
    recovery();
    configure('h7100, 6, 0, 7, 'h7100);
    retire_result(0);
    recovery();
    configure(64'hfffffffffffffffc, 8, 0, 7, 64'hfffffffffffffffc);
    retire_result(0);
    recovery();
    // Useful [VA,VA+4) fits, but the mandatory sixteen-byte transport span wraps.
    configure(64'hfffffffffffffffc, 4, 0, 7, 64'hfffffffffffffffc);
    retire_result(0);
    recovery();
    // Early last and missing last are protocol failures, never fabricated success.
    configure('h8000, 32, 0, 6, 'h8000);
    accept_request();
    response(0, 1);
    retire_result(1);
    recovery();
    configure('h8100, 16, 0, 6, 'h8100);
    accept_request();
    response(0, 0);
    no_terminal(8);
    response(4, 1);
    retire_result(1);
    recovery();
    `uvm_info("KERNEL_DMA_GATE", $sformatf(
              "%0d cases, %0d status/word checks; fault/cancel/drain/held terminal and recovery",
              cases,
              checked
              ), UVM_LOW)
    ctl.done = 1;
  end
endmodule
