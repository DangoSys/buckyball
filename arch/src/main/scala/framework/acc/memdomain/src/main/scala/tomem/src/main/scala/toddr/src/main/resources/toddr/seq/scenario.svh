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
  vif.io_dma_write_bits_vaddr = base;
  vif.io_dma_write_bits_len = length;
  vif.io_dma_write_valid = 1;
  do @(vif.sample); while (!vif.sample.io_dma_write_ready);
  @(negedge vif.clock);
  vif.io_dma_write_valid = 0;
  for (tick = 0; tick < 30000; tick++) begin
    vif.io_mappings_1_hit = 1;
    vif.io_mappings_1_error = 0;
    vif.io_mappings_1_pa = vif.io_queries_1_va;
    vif.io_dma_writeData_valid = sent < length / 16 && tick % 3 != 0;
    vif.io_dma_writeData_bits_data = data_at(base + sent * 16);
    vif.io_dma_writeData_bits_last = sent + 1 == length / 16;
    vif.io_axi_aw_ready = remaining == 0 && !pending_b && tick % 4 != 0;
    vif.io_axi_w_ready = remaining > 0 && tick % 5 != 0;
    vif.io_axi_b_valid = pending_b && tick % 3 == 0;
    vif.io_axi_b_bits_id = 1;
    vif.io_axi_b_bits_resp = bus_error ? 2 : 0;
    vif.io_dma_writeResult_ready = tick % 7 == 0;
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
          vif.sample.io_dma_writeResult_valid && vif.sample.io_dma_writeResult_bits_done == held_done && vif.sample.io_dma_writeResult_bits_fault_error == held_fault,
          "response changed under backpressure");
    held_response = vif.sample.io_dma_writeResult_valid && !vif.sample.io_dma_writeResult_ready;
    held_done = vif.sample.io_dma_writeResult_bits_done;
    held_fault = vif.sample.io_dma_writeResult_bits_fault_error;
    if (vif.sample.io_dma_writeData_valid && vif.sample.io_dma_writeData_ready) sent++;
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
    if (vif.sample.io_dma_writeResult_valid && vif.sample.io_dma_writeResult_ready) begin
      verify_contract(
          vif.sample.io_dma_writeResult_bits_fault_error == expected_error && vif.sample.io_dma_writeResult_bits_done == (expected_error == 0),
          "completion status mismatch");
      verify_contract(remaining == 0 && !pending_b,
                      "completed before accepted AXI traffic drained");
      if (expected_error == 0)
        verify_contract(sent == length / 16 && accepted_bytes == length,
                        "incomplete write transfer");
      @(negedge vif.clock);
      vif.io_dma_writeData_valid = 0;
      vif.io_axi_aw_ready = 0;
      vif.io_axi_w_ready = 0;
      vif.io_axi_b_valid = 0;
      vif.io_dma_writeResult_ready = 0;
      return;
    end
    @(negedge vif.clock);
  end
  `uvm_fatal("WRITE_DMA", "request timed out")
endtask
int overlap_cycles = 0;
task automatic run_read(int producer = 2);
  int pending, seen;
  longint unsigned address;
  pending = 0;
  seen = 0;
  vif.io_decisions_0_valid = 1;
  vif.io_decisions_2_valid = 1;
  vif.io_dma_read_bits_producer = producer;
  vif.io_dma_read_bits_transfer_vaddr = 'hc000;
  vif.io_dma_read_bits_transfer_len = 1024;
  vif.io_dma_read_bits_transfer_groups = 1;
  vif.io_dma_read_bits_transfer_stride = 1;
  vif.io_dma_read_valid = 1;
  do @(vif.sample); while (!vif.sample.io_dma_read_ready);
  @(negedge vif.clock);
  vif.io_dma_read_valid = 0;
  for (int tick = 0; tick < 10000; tick++) begin
    vif.io_mappings_0_hit = 1;
    vif.io_mappings_0_error = 0;
    vif.io_mappings_0_pa = vif.io_queries_0_va;
    vif.io_axi_ar_ready = pending == 0;
    vif.io_axi_r_valid = pending != 0;
    vif.io_axi_r_bits_data = data_at(address);
    vif.io_axi_r_bits_last = pending == 1;
    vif.io_dma_readResult_ready = tick % 3 != 0;
    @(vif.sample);
    if (vif.sample.io_dma_readBusy && vif.sample.io_dma_writeBusy) overlap_cycles++;
    if (vif.sample.io_axi_ar_valid && vif.sample.io_axi_ar_ready) begin
      pending = int'(vif.sample.io_axi_ar_bits_len) + 1;
      address = vif.sample.io_axi_ar_bits_addr;
    end
    if (vif.sample.io_axi_r_valid && vif.sample.io_axi_r_ready) begin
      pending--;
      address += 16;
    end
    if (vif.sample.io_dma_readResult_valid && vif.sample.io_dma_readResult_ready) begin
      verify_contract(vif.sample.io_dma_readResult_bits_data == data_at('hc000 + seen * 16
                      ) && vif.sample.io_dma_readResult_bits_fault_error == 0,
                      "concurrent read data mismatch");
      verify_contract(
          vif.sample.io_dma_readResult_bits_addrcounter == seen && vif.sample.io_dma_readResult_bits_last == (seen == 63),
          "concurrent read ordering mismatch");
      seen++;
      if (seen == 64) begin
        @(negedge vif.clock);
        vif.io_axi_r_valid = 0;
        vif.io_axi_ar_ready = 0;
        vif.io_dma_readResult_ready = 0;
        return;
      end
    end
    @(negedge vif.clock);
  end
  `uvm_fatal("DMA", "concurrent read timed out")
endtask

task execute();
  vif.initialize();
  repeat (3) @(vif.sample);
  @(negedge vif.clock);
  vif.reset = 0;
  vif.io_decisions_1_valid = 1;
  fork
    run_read();
    begin
      run_case('h1000, 16);
      run_case('h1ff0, 16 * 257);
      for (int offset = 1; offset < 16; offset++) run_case('h4ff0 + offset, 16 * 33);
      run_case('h9000, 0, 7);
      run_case('h9000, 17, 7);
      run_case(64'hfffffffffffffff1, 16, 7);
      run_case('ha003, 16 * 8, 2, 1);
      run_case('hb000, 32);
    end
  join
  run_read(0);
  verify_contract(overlap_cycles > 0, "read and write engines did not overlap");
  `uvm_info("WRITE_DMA", $sformatf("%0d cases %0d checks %0d bursts", cases, checks, bursts),
            UVM_LOW)


endtask
