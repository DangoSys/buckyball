module read_dma_tb;
  import uvm_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  ip_control_if ctl (clock);
  `include "read_dma_signals.svh"
ReadDma dut (
      `include "read_dma_ports.svh"
  );
  `include "read_dma_clocking.svh"
  int cases = 0, beats = 0, bursts = 0, checks = 0;
  function automatic bit [127:0] data_at(longint unsigned address);
    bit [127:0] value;
    for (int i = 0; i < 16; ++i) value[i*8+:8] = 8'((address + i) ^ ((address + i) >> 8));
    return value;
  endfunction
  task automatic check(bit good, string reason);
    checks++;
    if (!good) `uvm_fatal("READ_DMA", reason)
  endtask
  task automatic run_case(longint unsigned base, int length, int groups, int stride, bit mode2d = 0,
                          int pixel = 16, int width = 1, int source = 1, int expected_error = 0,
                          bit bus_error = 0, int cancel_after = 0);
    int count, seen, pending, issued, tick, fault_seen;
    longint unsigned read_address, expected_address, query_address;
    int query_bytes;
    bit [127:0] held_data;
    bit held, held_last;
    bit [ 3:0] held_error;
    bit [63:0] held_address;
    count = (length + 15) / 16;
    seen = 0;
    pending = 0;
    issued = 0;
    fault_seen = 0;
    held = 0;
    cases++;
    @(negedge clock);
    io_req_bits_vaddr = base;
    io_req_bits_len = length;
    io_req_bits_groups = groups;
    io_req_bits_stride = stride;
    io_req_bits_is_2d = mode2d;
    io_req_bits_pixel_bytes = pixel;
    io_req_bits_tile_width = width;
    io_req_bits_source_width = source;
    io_req_valid = 1;
    do @(sample); while (!sample.io_req_ready);
    @(negedge clock);
    io_req_valid = 0;
    for (tick = 0; tick < 20000; ++tick) begin
      io_mapping_hit = 1;
      io_mapping_error = 0;
      io_mapping_pa = io_query_va;
      io_axi_ar_ready = pending == 0 && tick % 3 != 0;
      io_axi_r_valid = pending != 0;
      io_axi_r_bits_data = data_at(read_address);
      io_axi_r_bits_last = pending == 1;
      io_axi_r_bits_resp = bus_error && issued == 1 ? 2 : 0;
      io_resp_ready = tick % 5 == 0;
      @(sample);
      if (held)
        check(
            sample.io_resp_valid && sample.io_resp_bits_data == held_data &&
                       sample.io_resp_bits_last == held_last && sample.io_resp_bits_fault_error == held_error &&
                       sample.io_resp_bits_fault_address == held_address,
            "response changed under backpressure");
      held = sample.io_resp_valid && !sample.io_resp_ready;
      held_data = sample.io_resp_bits_data;
      held_last = sample.io_resp_bits_last;
      held_error = sample.io_resp_bits_fault_error;
      held_address = sample.io_resp_bits_fault_address;
      if (sample.io_query_valid) begin
        check(
            !sample.io_query_write && sample.io_query_va[3:0] == 0 &&
              sample.io_query_bytes != 0 && sample.io_query_bytes % 16 == 0,
            "invalid prepared footprint");
        query_address = sample.io_mapping_pa;
        query_bytes   = sample.io_query_bytes;
      end
      if (sample.io_axi_ar_valid && sample.io_axi_ar_ready) begin
        check(
            sample.io_axi_ar_bits_addr == query_address &&
              (int'(sample.io_axi_ar_bits_len)+1)*16 == query_bytes,
            "burst differs from prepared footprint");
        check(sample.io_axi_ar_bits_size == 4 && sample.io_axi_ar_bits_burst == 1,
              "AXI request must be 128-bit INCR");
        check(
            sample.io_axi_ar_bits_addr[3:0] == 0 &&
              int'(sample.io_axi_ar_bits_addr[11:0]) + (int'(sample.io_axi_ar_bits_len)+1)*16 <= 4096,
            "burst alignment/page boundary");
        pending = int'(sample.io_axi_ar_bits_len) + 1;
        read_address = sample.io_axi_ar_bits_addr;
        bursts++;
      end
      if (sample.io_axi_r_valid && sample.io_axi_r_ready) begin
        pending--;
        issued++;
        read_address += 16;
      end
      if (sample.io_resp_valid && sample.io_resp_ready) begin
        if (sample.io_resp_bits_fault_error != 0) begin
          check(sample.io_resp_bits_fault_error == expected_error && sample.io_resp_bits_last,
                "wrong terminal fault");
          check(pending == 0, "fault completed before accepted burst drained");
          fault_seen++;
        end else begin
          check(expected_error == 0 || bus_error, "shape failure dispatched data");
          expected_address = mode2d ? base + (seen/width)*source*pixel + (seen%width)*pixel :
                                     base + (seen/groups)*groups*stride*16 + (seen%groups)*16;
          check(sample.io_resp_bits_data == data_at(expected_address),
                "data/stride/unaligned assembly mismatch");
          check(sample.io_resp_bits_addrcounter == seen, "beat index mismatch");
          if (!bus_error) check(sample.io_resp_bits_last == (seen + 1 == count), "last mismatch");
          seen++;
          beats++;
        end
        if (sample.io_resp_bits_last) begin
          check(fault_seen == (expected_error != 0), "missing/unexpected terminal error");
          if (!expected_error) check(seen == count, "incomplete response");
          @(negedge clock);
          io_resp_ready   = 0;
          io_axi_r_valid  = 0;
          io_axi_ar_ready = 0;
          return;
        end
      end
      @(negedge clock);
      if (cancel_after != 0 && tick == cancel_after) begin
        check(issued == 0, "reset fixture must precede AXI issue");
        reset = 1;
        io_axi_r_valid = 0;
        io_axi_ar_ready = 0;
        repeat (3) @(sample);
        @(negedge clock);
        reset = 0;
        repeat (4) begin
          @(sample);
          check(!sample.io_resp_valid && !sample.io_axi_ar_valid && !sample.io_busy,
                "reset leaked old work");
        end
        return;
      end
    end
    `uvm_fatal("READ_DMA", "request timed out")
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 2ms);
    run_test("protocol_test");
  end
  initial begin
    `include "read_dma_init.svh"
    wait (ctl.start);
    repeat (4) @(sample);
    @(negedge clock);
    reset = 0;
    io_decisionValid = 1;
    run_case('h1000, 16, 1, 1);
    run_case('h1ff0, 16 * 257, 1, 1);
    run_case('h3003, 53, 3, 5);
    run_case('h4ff9, 17, 1, 1);
    run_case('h6000, 16 * 19, 3, 7);
    run_case('h8000, 16 * 64, 63, 524287);
    run_case('h8000, 16 * 63, 63, 524287);
    run_case('h9000, 16 * 20, 1, 1, 1, 7, 3, 7);
    run_case('haff0, 16 * 20, 1, 1, 1, 16, 3, 7);
    run_case('hb000, 16 * 17, 1, 1, 1, 1023, 15, 1023);
    run_case(64'hfffffffffffffff0, 16, 1, 1, 0, 16, 1, 1, 0);
    run_case(64'hfffffffffffffff1, 16, 1, 1, 0, 16, 1, 1, 7);
    run_case(64'hfffffff000000000, 16 * 65536, 63, 524287, 0, 16, 1, 1, 7);
    run_case(64'hfffffffffff00000, 16 * 65536, 1, 1, 1, 1023, 15, 1023, 7);
    run_case('hc000, 0, 1, 1, 0, 16, 1, 1, 7);
    run_case('hc000, 16, 0, 1, 0, 16, 1, 1, 7);
    run_case('hc000, 16, 1, 1, 1, 0, 3, 3, 7);
    run_case('hd000, 64, 1, 1, 0, 16, 1, 1, 2, 1);
    run_case('he000, 256, 3, 1, 0, 16, 1, 1, 0, 0, 5);
    run_case('hf000, 32, 1, 1);
    `uvm_info("READ_DMA", $sformatf(
              "%0d cases %0d checks %0d beats %0d bursts", cases, checks, beats, bursts), UVM_LOW)
    ctl.done = 1;
  end
endmodule
