module storer_ack_tb;
  import uvm_pkg::*;
  import ack_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 0;
  always #5 clock = ~clock;
  ip_control_if ctl (clock);
  `include "storer_ack_signals.svh"
stream_if #(
      .WIDTH($bits(
          io_cmdResp_bits_rob_id
      ) + $bits(
          io_cmdResp_bits_is_sub
      ) + $bits(
          io_cmdResp_bits_sub_rob_id
      ) + $bits(
          io_cmdResp_bits_fault_error
      ) + $bits(
          io_cmdResp_bits_fault_address
      ))
  ) sink_if (
      clock,
      reset
  );
  assign sink_if.valid = io_cmdResp_valid;
  assign sink_if.ready = io_cmdResp_ready;
  assign sink_if.bits = {
    io_cmdResp_bits_rob_id,
    io_cmdResp_bits_is_sub,
    io_cmdResp_bits_sub_rob_id,
    io_cmdResp_bits_fault_error,
    io_cmdResp_bits_fault_address
  };
  MemStorer dut (
      `include "storer_ack_ports.svh"
  );
  stream_if #(
      .WIDTH($bits(
          io_cmdReq_bits_cmd_special
      ) + $bits(
          io_cmdReq_bits_cmd_mem_addr
      ) + $bits(
          io_cmdReq_bits_cmd_iter
      ) + $bits(
          io_cmdReq_bits_cmd_bank_id
      ))
  ) source_if (
      clock,
      reset
  );
  assign source_if.valid = io_cmdReq_valid;
  assign source_if.ready = io_cmdReq_ready;
  assign source_if.bits = {
    io_cmdReq_bits_cmd_special,
    io_cmdReq_bits_cmd_mem_addr,
    io_cmdReq_bits_cmd_iter,
    io_cmdReq_bits_cmd_bank_id
  };
  stream_if #(
      .WIDTH($bits(
          io_footprint_rob_id
      ) + $bits(
          io_footprint_is_sub
      ) + $bits(
          io_footprint_sub_rob_id
      ) + $bits(
          io_footprint_baseVA
      ) + $bits(
          io_footprint_rows
      ) + $bits(
          io_footprint_columns
      ) + $bits(
          io_footprint_spanBytes
      ) + $bits(
          io_footprint_columnStride
      ) + $bits(
          io_footprint_rowStride
      ) + $bits(
          io_footprint_write
      ) + $bits(
          io_footprint_fault_error
      ) + $bits(
          io_footprint_fault_address
      ))
  ) footprint_if (
      clock,
      reset
  );
  assign footprint_if.valid = io_footprint_valid;
  assign footprint_if.ready = io_cmdResp_valid && io_cmdResp_ready;
  assign footprint_if.bits = {
    io_footprint_rob_id,
    io_footprint_is_sub,
    io_footprint_sub_rob_id,
    io_footprint_baseVA,
    io_footprint_rows,
    io_footprint_columns,
    io_footprint_spanBytes,
    io_footprint_columnStride,
    io_footprint_rowStride,
    io_footprint_write,
    io_footprint_fault_error,
    io_footprint_fault_address
  };
  chandle model;
  int completions = 0, store_cases = 0, descriptor_count = 0, data_count = 0;
  logic [127:0] expected;
  longint unsigned descriptor_address;
  int descriptor_bytes;
  logic [127:0] descriptor_data[$];
  function automatic logic [127:0] row_data(int row);
    logic [127:0] data;
    for (int j = 0; j < 16; j++) data[j*8+:8] = (row * 37 + j * 11 + 3) & 255;
    return data;
  endfunction
  task automatic check(bit condition, string message);
    if (!condition) `uvm_fatal("ACK_CONTRACT", message)
  endtask
  task automatic no_completion(int cycles);
    repeat (cycles) begin
      @(posedge clock);
      #1;
      check(!io_cmdResp_valid, "cmdResp.valid asserted before final descriptor ACK");
      check(!io_cmdReq_ready, "accepted next command before completion");
      check(completions == 0, "unexpected completion count before descriptor ACK");
    end
  endtask
  task automatic complete_command(int error = 0, longint unsigned address = 0);
    wait (io_cmdResp_valid);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(
          io_cmdResp_valid && io_cmdResp_bits_rob_id == 3 && io_cmdResp_bits_is_sub == 1 && io_cmdResp_bits_sub_rob_id == 2,
          "completion changed under backpressure");
      check(io_cmdResp_bits_fault_error == error && io_cmdResp_bits_fault_address == address,
            "completion fault changed under backpressure");
      check(!io_cmdReq_ready, "command accepted while completion stalled");
    end
    @(negedge clock);
    io_cmdResp_ready = 1;
    @(posedge clock);
    check(io_cmdResp_valid, "completion lost before handshake");
    completions++;
    @(negedge clock);
    io_cmdResp_ready = 0;
    repeat (8) begin
      @(posedge clock);
      #1;
      check(!io_cmdResp_valid, "duplicate completion");
    end
    check(completions == 1 && io_cmdReq_ready, "completion count or idle readiness wrong");
  endtask
  task automatic begin_store(longint unsigned address, int rows, int groups = 1, int stride = 1);
    completions = 0;
    store_cases++;
    descriptor_count = 0;
    data_count = 0;
    @(negedge clock);
    io_cmdReq_bits_cmd_mem_addr = address;
    io_cmdReq_bits_cmd_iter = rows;
    io_query_group_count = groups;
    io_cmdReq_bits_cmd_special = 64'(stride) << 39;
    io_dmaResp_bits_fault_error = 0;
    io_dmaResp_bits_fault_address = 0;
    io_cmdReq_valid = 1;
    do @(posedge clock); while (!io_cmdReq_ready);
    @(negedge clock);
    io_cmdReq_valid = 0;
  endtask
  task automatic accept_descriptor(longint unsigned address, int bytes);
    wait (io_dmaReq_valid);
    repeat (4) begin
      @(posedge clock);
      #1;
      check(io_dmaReq_valid && io_dmaReq_bits_vaddr == address && io_dmaReq_bits_len == bytes,
            "whole-descriptor address/length changed under backpressure");
      check(!io_cmdResp_valid, "descriptor offer released Core completion");
    end
    @(negedge clock);
    io_dmaReq_ready = 1;
    @(posedge clock);
    check(io_dmaReq_valid, "missing descriptor handshake");
    descriptor_address = io_dmaReq_bits_vaddr;
    descriptor_bytes   = io_dmaReq_bits_len;
    descriptor_data.delete();
    descriptor_count++;
    @(negedge clock);
    io_dmaReq_ready = 0;
  endtask
  task automatic bank_and_data(int row, int group, logic [127:0] data, bit last);
    wait (io_bankRead_io_req_valid);
    check(
        io_bankRead_io_req_bits_addr == row && io_bankRead_group_id == group &&
          io_bankRead_bank_id == 1 && io_bankRead_rob_id == 3,
        "bank row/group/tag wrong");
    repeat (2) begin
      @(posedge clock);
      #1;
      check(
          io_bankRead_io_req_valid && io_bankRead_io_req_bits_addr == row &&
            io_bankRead_group_id == group,
          "bank read changed under backpressure");
    end
    @(negedge clock);
    io_bankRead_io_req_ready = 1;
    @(posedge clock);
    check(io_bankRead_io_req_valid, "missing bank read handshake");
    @(negedge clock);
    io_bankRead_io_req_ready = 0;
    no_completion(2);
    @(negedge clock);
    io_bankRead_io_resp_bits_data = data;
    io_bankRead_io_resp_valid = 1;
    do @(posedge clock); while (!io_bankRead_io_resp_ready);
    @(negedge clock);
    io_bankRead_io_resp_valid = 0;
    wait (io_dmaData_valid);
    repeat (3) begin
      @(posedge clock);
      #1;
      check(io_dmaData_valid && io_dmaData_bits_data === data && io_dmaData_bits_last == last,
            "raw 128-bit bank stream changed under backpressure or was shifted for alignment");
      check(!io_dmaReq_valid, "duplicate descriptor emitted during its stream");
      check(!io_cmdResp_valid, "Core completion preceded final descriptor ACK");
    end
    @(negedge clock);
    io_dmaData_ready = 1;
    @(posedge clock);
    check(io_dmaData_valid, "missing raw DMA data handshake");
    descriptor_data.push_back(io_dmaData_bits_data);
    data_count++;
    @(negedge clock);
    io_dmaData_ready = 0;
  endtask
  task automatic feed_descriptor(longint unsigned base, int rows, int groups = 1, int stride = 1,
                                 int first_row = 0, int descriptor_rows = 0);
    int count;
    count = descriptor_rows == 0 ? rows : descriptor_rows;
    accept_descriptor(base + first_row * groups * stride * 16, count * groups * 16);
    check(
        io_footprint_valid && io_footprint_baseVA == base && io_footprint_rows == rows &&
          io_footprint_columns == 1 && io_footprint_spanBytes == groups * 16 &&
          io_footprint_columnStride == 0 && io_footprint_rowStride == groups * stride * 16 &&
          io_footprint_fault_error == 0 && io_footprint_write,
        "footprint differs from the complete store command");
    for (int row = first_row; row < first_row + count; row++)
      for (int group = 0; group < groups; group++)
        bank_and_data(row, group, row_data(row * groups + group),
                      row == first_row + count - 1 && group == groups - 1);
    check(descriptor_data.size() * 16 == descriptor_bytes, "stream length differs from descriptor");
  endtask
  task automatic dma_ack(bit done, int error = 0, longint unsigned address = 0);
    no_completion(32);
    if (done && error == 0)
      foreach (descriptor_data[index])
        ack_ref_write(model, descriptor_address + index * 16, descriptor_data[index][63:0],
                      descriptor_data[index][127:64], 'hffff);
    @(negedge clock);
    io_dmaResp_bits_done = done;
    io_dmaResp_bits_fault_error = error;
    io_dmaResp_bits_fault_address = address;
    io_dmaResp_valid = 1;
    do @(posedge clock); while (!io_dmaResp_ready);
    @(negedge clock);
    io_dmaResp_valid = 0;
  endtask
  task automatic no_future_requests();
    repeat (8) begin
      @(posedge clock);
      #1;
      check(!io_dmaReq_valid && !io_dmaData_valid && !io_bankRead_io_req_valid,
            "failed final ACK launched residual DMA data/descriptor or a future bank read");
    end
  endtask
  task automatic recovery(longint unsigned address);
    begin_store(address, 1);
    feed_descriptor(address, 1);
    dma_ack(1);
    complete_command();
    check(descriptor_count == 1 && data_count == 1,
          "recovery reused previous row/descriptor state");
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 100us);
    run_test("protocol_test");
  end
  initial begin
    logic [127:0] partial;
    `include "storer_ack_init.svh"
    wait (ctl.start);
    model = ack_ref_create();
    @(negedge clock);
    reset = 1;
    repeat (4) @(posedge clock);
    @(negedge clock);
    reset = 0;
    io_cmdReq_bits_cmd_bank_id = 1;
    io_cmdReq_bits_rob_id = 3;
    io_cmdReq_bits_is_sub = 1;
    io_cmdReq_bits_sub_rob_id = 2;
    io_cmdReq_bits_cmd_is_store = 1;
    begin_store('h1000, 4);
    feed_descriptor('h1000, 4);
    check(descriptor_count == 1 && data_count == 4, "stride-one rows were not one long descriptor");
    dma_ack(1);
    complete_command();
    for (int row = 0; row < 4; row++) begin
      expected = row_data(row);
      check(ack_ref_check(model, 'h1000 + row * 16, expected[63:0], expected[127:64]) == 1,
            "DDR golden after final completion failed");
    end
    // The packing/partial-burst failure belongs to WriteDma, not MemStorer.
    begin_store('h4003, 4);
    feed_descriptor('h4003, 4);
    dma_ack(0, 3, 'h4000);
    no_future_requests();
    complete_command(3, 'h4000);
    recovery('h4100);
    // Preserve already committed bytes when a later DMA segment faults.
    begin_store('h5003, 4);
    feed_descriptor('h5003, 4);
    partial = ~row_data(0);
    ack_ref_write(model, 'h5003, partial[63:0], partial[127:64], 'hffff);
    expected = row_data(0);
    ack_ref_write(model, 'h5003, expected[63:0], expected[127:64], 'h1fff);
    partial[103:0] = expected[103:0];
    dma_ack(0, 4, 'h5010);
    no_future_requests();
    complete_command(4, 'h5010);
    check(ack_ref_check(model, 'h5003, partial[63:0], partial[127:64]) == 1,
          "later-segment fault rolled back prior committed bytes");
    recovery('h5100);
    begin_store('h6000, 1);
    feed_descriptor('h6000, 1);
    dma_ack(0);
    no_future_requests();
    complete_command(6, 'h6000);
    recovery('h6100);
    // The descriptor keeps the unaligned VA, and data is sent unshifted once.
    begin_store('h7003, 1);
    feed_descriptor('h7003, 1);
    dma_ack(1);
    complete_command();
    expected = row_data(0);
    check(descriptor_count == 1 && data_count == 1 && ack_ref_check(
          model, 'h7003, expected[63:0], expected[127:64]) == 1,
          "unaligned store changed raw bytes or emitted a second descriptor");
    // Each strided row has one groups*16-byte descriptor; fault suppresses row1.
    begin_store('h8000, 2, 2, 3);
    feed_descriptor('h8000, 2, 2, 3, 0, 1);
    dma_ack(0, 3, 'h8000);
    no_future_requests();
    complete_command(3, 'h8000);
    check(descriptor_count == 1 && data_count == 2, "strided row did not group its raw bank beats");
    recovery('h8100);
    check(store_cases == 10, "original store/recovery case count changed");
    ack_ref_destroy(model);
    `uvm_info("ACK_GATE",
              "10 existing cases: descriptor-final ACK delay, raw128 stable stream, typed fault/protocol fault, strided drain, stable completion and recovery",
              UVM_LOW)
    ctl.done = 1;
  end
endmodule
