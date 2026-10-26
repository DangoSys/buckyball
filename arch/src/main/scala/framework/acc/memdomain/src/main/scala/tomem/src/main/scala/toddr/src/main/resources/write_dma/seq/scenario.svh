task automatic run_case(longint unsigned base, int length, int expected_error = 0,
                        bit bus_error = 0);
  int sent, remaining, accepted_bytes, tick;
  longint unsigned write_address;
  bit pending_b, held;
  bit [127:0] held_data;
  bit [15:0] held_mask;
  bit held_last;
  bit held_response, held_done;
  bit [3:0] held_fault;
  sent = 0;
  remaining = 0;
  accepted_bytes = 0;
  pending_b = 0;
  held = 0;
  held_response = 0;
  cases++;
  behavior.sample(base, length, expected_error, bus_error, 0, 0);
  vif.io_req_bits_vaddr = base;
  vif.io_req_bits_len = length;
  vif.io_req_valid = 1;
  do @(vif.sample); while (!vif.sample.io_req_ready);
  @(negedge vif.clock);
  vif.io_req_valid = 0;
  for (tick = 0; tick < 30000; tick++) begin
    vif.io_mapping_hit = 1;
    vif.io_mapping_error = 0;
    vif.io_mapping_pa = vif.io_query_va;
    vif.io_data_valid = sent < length / 16 && tick % 3 != 0;
    vif.io_data_bits_data = data_at(base + sent * 16);
    vif.io_data_bits_last = sent + 1 == length / 16;
    vif.io_axi_aw_ready = remaining == 0 && !pending_b && tick % 4 != 0;
    vif.io_axi_w_ready = remaining > 0 && tick % 5 != 0;
    vif.io_axi_b_valid = pending_b && tick % 3 == 0;
    vif.io_axi_b_bits_id = 1;
    vif.io_axi_b_bits_resp = bus_error ? 2 : 0;
    vif.io_resp_ready = tick % 7 == 0;
    @(vif.sample);
    if (held)
      verify_contract(
          vif.sample.io_axi_w_valid && vif.sample.io_axi_w_bits_data == held_data && vif.sample.io_axi_w_bits_strb == held_mask && vif.sample.io_axi_w_bits_last == held_last,
          "W changed under backpressure");
    held = vif.sample.io_axi_w_valid && !vif.sample.io_axi_w_ready;
    held_data = vif.sample.io_axi_w_bits_data;
    held_mask = vif.sample.io_axi_w_bits_strb;
    held_last = vif.sample.io_axi_w_bits_last;
    if (held_response)
      verify_contract(
          vif.sample.io_resp_valid && vif.sample.io_resp_bits_done == held_done && vif.sample.io_resp_bits_fault_error == held_fault,
          "response changed under backpressure");
    held_response = vif.sample.io_resp_valid && !vif.sample.io_resp_ready;
    held_done = vif.sample.io_resp_bits_done;
    held_fault = vif.sample.io_resp_bits_fault_error;
    if (vif.sample.io_data_valid && vif.sample.io_data_ready) sent++;
    if (vif.sample.io_axi_aw_valid && vif.sample.io_axi_aw_ready) begin
      remaining = int'(vif.sample.io_axi_aw_bits_len) + 1;
      write_address = vif.sample.io_axi_aw_bits_addr;
      verify_contract(
          vif.sample.io_axi_aw_bits_id == 1 && vif.sample.io_axi_aw_bits_size == 4 && vif.sample.io_axi_aw_bits_burst == 1,
          "AW format mismatch");
      verify_contract(write_address % 16 == 0 && (write_address % 4096) + remaining * 16 <= 4096,
                      "AW crosses a page");
      bursts++;
    end
    if (vif.sample.io_axi_w_valid && vif.sample.io_axi_w_ready) begin
      verify_contract(remaining > 0 && vif.sample.io_axi_w_bits_last == (remaining == 1),
                      "WLAST mismatch");
      for (int i = 0; i < 16; i++) begin
        verify_contract(
            vif.sample.io_axi_w_bits_strb[i] == (write_address+i >= base && write_address+i < base+length),
            "byte mask changed the transfer footprint");
        if (vif.sample.io_axi_w_bits_strb[i]) begin
          verify_contract(vif.sample.io_axi_w_bits_data[8*i+:8] == 8'(dma_ref_byte(write_address + i
                          )), "unaligned write data mismatch");
          accepted_bytes++;
        end
      end
      remaining--;
      write_address += 16;
      if (remaining == 0) pending_b = 1;
    end
    if (vif.sample.io_axi_b_valid && vif.sample.io_axi_b_ready) pending_b = 0;
    if (vif.sample.io_resp_valid && vif.sample.io_resp_ready) begin
      verify_contract(
          vif.sample.io_resp_bits_fault_error == expected_error && vif.sample.io_resp_bits_done == (expected_error == 0),
          "completion status mismatch");
      verify_contract(remaining == 0 && !pending_b,
                      "completed before accepted AXI traffic drained");
      if (expected_error == 0)
        verify_contract(sent == length / 16 && accepted_bytes == length,
                        "incomplete write transfer");
      @(negedge vif.clock);
      vif.io_data_valid   = 0;
      vif.io_axi_aw_ready = 0;
      vif.io_axi_w_ready  = 0;
      vif.io_axi_b_valid  = 0;
      vif.io_resp_ready   = 0;
      return;
    end
    @(negedge vif.clock);
  end
  `uvm_fatal("WRITE_DMA", "request timed out")
endtask

task execute();
  vif.initialize();
  repeat (3) @(vif.sample);
  @(negedge vif.clock);
  vif.reset = 0;
  vif.io_decisionValid = 1;
  run_case('h1000, 16);
  run_case('h1ff0, 16 * 257);
  for (int offset = 1; offset < 16; offset++) run_case('h4ff0 + offset, 16 * 33);
  run_case('h9000, 0, 7);
  run_case('h9000, 17, 7);
  run_case(64'hfffffffffffffff1, 16, 7);
  run_case('ha003, 16 * 8, 2, 1);
  run_case('hb000, 32);
  `uvm_info("WRITE_DMA", $sformatf("%0d cases %0d checks %0d bursts", cases, checks, bursts),
            UVM_LOW)


endtask
