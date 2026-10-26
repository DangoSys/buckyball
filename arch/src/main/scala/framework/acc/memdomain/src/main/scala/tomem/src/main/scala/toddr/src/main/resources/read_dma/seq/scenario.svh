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
  behavior.sample(base, length, expected_error, bus_error, mode2d, cancel_after != 0);
  @(negedge vif.clock);
  vif.io_req_bits_vaddr = base;
  vif.io_req_bits_len = length;
  vif.io_req_bits_groups = groups;
  vif.io_req_bits_stride = stride;
  vif.io_req_bits_is_2d = mode2d;
  vif.io_req_bits_pixel_bytes = pixel;
  vif.io_req_bits_core_width = width;
  vif.io_req_bits_source_width = source;
  vif.io_req_valid = 1;
  do @(vif.sample); while (!vif.sample.io_req_ready);
  @(negedge vif.clock);
  vif.io_req_valid = 0;
  for (tick = 0; tick < 20000; ++tick) begin
    vif.io_mapping_hit = 1;
    vif.io_mapping_error = 0;
    vif.io_mapping_pa = vif.io_query_va;
    vif.io_axi_ar_ready = pending == 0 && tick % 3 != 0;
    vif.io_axi_r_valid = pending != 0;
    vif.io_axi_r_bits_data = data_at(read_address);
    vif.io_axi_r_bits_last = pending == 1;
    vif.io_axi_r_bits_resp = bus_error && issued == 1 ? 2 : 0;
    vif.io_resp_ready = tick % 5 == 0;
    @(vif.sample);
    if (held)
      verify_contract(
          vif.sample.io_resp_valid && vif.sample.io_resp_bits_data == held_data &&
                       vif.sample.io_resp_bits_last == held_last && vif.sample.io_resp_bits_fault_error == held_error &&
                       vif.sample.io_resp_bits_fault_address == held_address,
          "response changed under backpressure");
    held = vif.sample.io_resp_valid && !vif.sample.io_resp_ready;
    held_data = vif.sample.io_resp_bits_data;
    held_last = vif.sample.io_resp_bits_last;
    held_error = vif.sample.io_resp_bits_fault_error;
    held_address = vif.sample.io_resp_bits_fault_address;
    if (vif.sample.io_query_valid) begin
      verify_contract(
          !vif.sample.io_query_write && vif.sample.io_query_va[3:0] == 0 &&
              vif.sample.io_query_bytes != 0 && vif.sample.io_query_bytes % 16 == 0,
          "invalid prepared footprint");
      query_address = vif.sample.io_mapping_pa;
      query_bytes   = vif.sample.io_query_bytes;
    end
    if (vif.sample.io_axi_ar_valid && vif.sample.io_axi_ar_ready) begin
      verify_contract(
          vif.sample.io_axi_ar_bits_addr == query_address &&
              (int'(vif.sample.io_axi_ar_bits_len)+1)*16 == query_bytes,
          "burst differs from prepared footprint");
      verify_contract(vif.sample.io_axi_ar_bits_size == 4 && vif.sample.io_axi_ar_bits_burst == 1,
                      "AXI request must be 128-bit INCR");
      verify_contract(
          vif.sample.io_axi_ar_bits_addr[3:0] == 0 &&
              int'(vif.sample.io_axi_ar_bits_addr[11:0]) + (int'(vif.sample.io_axi_ar_bits_len)+1)*16 <= 4096,
          "burst alignment/page boundary");
      pending = int'(vif.sample.io_axi_ar_bits_len) + 1;
      read_address = vif.sample.io_axi_ar_bits_addr;
      bursts++;
    end
    if (vif.sample.io_axi_r_valid && vif.sample.io_axi_r_ready) begin
      pending--;
      issued++;
      read_address += 16;
    end
    if (vif.sample.io_resp_valid && vif.sample.io_resp_ready) begin
      if (vif.sample.io_resp_bits_fault_error != 0) begin
        verify_contract(
            vif.sample.io_resp_bits_fault_error == expected_error && vif.sample.io_resp_bits_last,
            "wrong terminal fault");
        verify_contract(pending == 0, "fault completed before accepted burst drained");
        fault_seen++;
      end else begin
        verify_contract(expected_error == 0 || bus_error, "shape failure dispatched data");
        expected_address = mode2d ? base + (seen/width)*source*pixel + (seen%width)*pixel :
                                     base + (seen/groups)*groups*stride*16 + (seen%groups)*16;
        verify_contract(vif.sample.io_resp_bits_data == data_at(expected_address),
                        "data/stride/unaligned assembly mismatch");
        verify_contract(vif.sample.io_resp_bits_addrcounter == seen, "beat index mismatch");
        if (!bus_error)
          verify_contract(vif.sample.io_resp_bits_last == (seen + 1 == count), "last mismatch");
        seen++;
        beats++;
      end
      if (vif.sample.io_resp_bits_last) begin
        verify_contract(fault_seen == (expected_error != 0), "missing/unexpected terminal error");
        if (!expected_error) verify_contract(seen == count, "incomplete response");
        @(negedge vif.clock);
        vif.io_resp_ready   = 0;
        vif.io_axi_r_valid  = 0;
        vif.io_axi_ar_ready = 0;
        return;
      end
    end
    @(negedge vif.clock);
    if (cancel_after != 0 && tick == cancel_after) begin
      verify_contract(issued == 0, "vif.reset fixture must precede AXI issue");
      vif.reset = 1;
      vif.io_axi_r_valid = 0;
      vif.io_axi_ar_ready = 0;
      repeat (3) @(vif.sample);
      @(negedge vif.clock);
      vif.reset = 0;
      repeat (4) begin
        @(vif.sample);
        verify_contract(
            !vif.sample.io_resp_valid && !vif.sample.io_axi_ar_valid && !vif.sample.io_busy,
            "vif.reset leaked old work");
      end
      return;
    end
  end
  `uvm_fatal("READ_DMA", "request timed out")
endtask

task execute();
  vif.initialize();
  repeat (4) @(vif.sample);
  @(negedge vif.clock);
  vif.reset = 0;
  vif.io_decisionValid = 1;
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
  `uvm_info("READ_DMA", $sformatf("%0d cases %0d checks %0d beats %0d bursts", cases, checks,
                                  beats, bursts), UVM_LOW)


endtask
