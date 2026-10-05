module loader_ack_tb;
  import uvm_pkg::*;
  import ack_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 0;
  always #5 clock = ~clock;
  ip_control_if ctl (clock);
  `include "loader_ack_signals.svh"
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
  MemLoader dut (
      `include "loader_ack_ports.svh"
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
  int completions = 0;
  logic [127:0] expected, observed;
  longint unsigned observed_addr;
  int unsigned observed_mask;
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
      check(!io_cmdResp_valid, "cmdResp.valid asserted before final ACK");
      check(!io_cmdReq_ready, "accepted next command before completion");
      check(completions == 0, "unexpected completion count before ACK");
    end
  endtask
  task automatic complete_command(int error = 0, longint unsigned address = 0);
    wait (io_cmdResp_valid);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(
          io_cmdResp_valid && io_cmdResp_bits_rob_id==3 && io_cmdResp_bits_is_sub==1 && io_cmdResp_bits_sub_rob_id==2,
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
  task automatic begin_load(longint unsigned address, int rows, int groups = 1, int stride = 1);
    completions = 0;
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
    wait (io_dmaReq_valid);
    check(
        io_footprint_valid && io_footprint_baseVA==address && io_footprint_rows==rows && io_footprint_columns==1 && io_footprint_spanBytes==groups*16 && io_footprint_columnStride==0 && io_footprint_rowStride==groups*stride*16 && io_footprint_fault_error==0 && io_footprint_write==0,
        "production footprint differs from actual command");
    check(io_dmaReq_bits_vaddr == address && io_dmaReq_bits_len == rows * groups * 16,
          "followup load DMA shape wrong");
    @(negedge clock);
    io_dmaReq_ready = 1;
    @(posedge clock);
    check(io_dmaReq_valid, "followup DMA request missing");
    @(negedge clock);
    io_dmaReq_ready = 0;
  endtask
  task automatic read_beat(int row, bit last, int error = 0, longint unsigned address = 0);
    @(negedge clock);
    io_dmaResp_bits_data = row_data(row);
    io_dmaResp_bits_last = last;
    io_dmaResp_bits_addrcounter = row;
    io_dmaResp_bits_fault_error = error;
    io_dmaResp_bits_fault_address = address;
    io_dmaResp_valid = 1;
    do @(posedge clock); while (!io_dmaResp_ready);
    @(negedge clock);
    io_dmaResp_valid = 0;
  endtask
  task automatic bank_ack(int row, bit ok);
    wait (io_bankWrite_io_req_valid);
    check(io_bankWrite_io_req_bits_addr == row && io_bankWrite_io_req_bits_data === row_data(row),
          "followup bank data/address wrong");
    @(negedge clock);
    io_bankWrite_io_req_ready = 1;
    @(posedge clock);
    check(io_bankWrite_io_req_valid, "followup bank request missing");
    @(negedge clock);
    io_bankWrite_io_req_ready = 0;
    no_completion(16);
    @(negedge clock);
    io_bankWrite_io_resp_bits_ok = ok;
    io_bankWrite_io_resp_valid   = 1;
    do @(posedge clock); while (!io_bankWrite_io_resp_ready);
    @(negedge clock);
    io_bankWrite_io_resp_valid = 0;
  endtask
  task automatic no_bank_write(int cycles);
    repeat (cycles) begin
      @(posedge clock);
      #1;
      check(!io_bankWrite_io_req_valid, "bank writes continued after first fault");
    end
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 100us);
    run_test("protocol_test");
  end
  initial begin
    `include "loader_ack_init.svh"
    wait (ctl.start);
    model = ack_ref_create();
    @(negedge clock);
    reset = 1;
    repeat (4) @(posedge clock);
    @(negedge clock);
    reset = 0;
    io_query_group_count = 1;
    io_cmdReq_bits_cmd_mem_addr = 64'h1000;
    io_cmdReq_bits_cmd_iter = 4;
    io_cmdReq_bits_cmd_bank_id = 1;
    io_cmdReq_bits_cmd_special = 64'd1 << 39;
    io_cmdReq_bits_rob_id = 3;
    io_cmdReq_bits_is_sub = 1;
    io_cmdReq_bits_sub_rob_id = 2;
    io_cmdReq_bits_cmd_is_load = 1;
    io_cmdReq_valid = 1;
    do @(posedge clock); while (!io_cmdReq_ready);
    @(negedge clock);
    io_cmdReq_valid = 0;
    wait (io_dmaReq_valid);
    check(
        io_footprint_valid && io_footprint_baseVA=='h1000 && io_footprint_rows==4 && io_footprint_columns==1 && io_footprint_spanBytes==16 && io_footprint_rowStride==16 && !io_footprint_write && io_footprint_fault_error==0,
        "initial loader footprint wrong");
    check(
        io_dmaReq_bits_vaddr==64'h1000 && io_dmaReq_bits_len==64 && io_dmaReq_bits_stride==1 && io_dmaReq_bits_groups==1,
        "load DMA request wrong");
    repeat (3) begin
      @(posedge clock);
      #1;
      check(io_dmaReq_valid && io_dmaReq_bits_vaddr == 64'h1000,
            "DMA request changed while stalled");
    end
    @(negedge clock);
    io_dmaReq_ready = 1;
    @(posedge clock);
    check(io_dmaReq_valid, "missing DMA handshake");
    @(negedge clock);
    io_dmaReq_ready = 0;
    for (int row = 0; row < 4; row++) begin
      expected = row_data(row);
      io_dmaResp_bits_data = expected;
      io_dmaResp_bits_last = (row == 3);
      io_dmaResp_bits_addrcounter = row;
      io_dmaResp_valid = 1;
      do @(posedge clock); while (!io_dmaResp_ready);
      @(negedge clock);
      io_dmaResp_valid = 0;
      wait (io_bankWrite_io_req_valid);
      check(
          io_bankWrite_io_req_bits_addr==row && io_bankWrite_io_req_bits_data===expected && (&{io_bankWrite_io_req_bits_mask_0, io_bankWrite_io_req_bits_mask_1, io_bankWrite_io_req_bits_mask_2, io_bankWrite_io_req_bits_mask_3, io_bankWrite_io_req_bits_mask_4, io_bankWrite_io_req_bits_mask_5, io_bankWrite_io_req_bits_mask_6, io_bankWrite_io_req_bits_mask_7, io_bankWrite_io_req_bits_mask_8, io_bankWrite_io_req_bits_mask_9, io_bankWrite_io_req_bits_mask_10, io_bankWrite_io_req_bits_mask_11, io_bankWrite_io_req_bits_mask_12, io_bankWrite_io_req_bits_mask_13, io_bankWrite_io_req_bits_mask_14, io_bankWrite_io_req_bits_mask_15}) && io_bankWrite_bank_id==1 && io_bankWrite_rob_id==3 && io_bankWrite_group_id==0,
          "bank write address/data/mask/tag wrong");
      repeat (2) begin
        @(posedge clock);
        #1;
        check(io_bankWrite_io_req_valid && io_bankWrite_io_req_bits_data === expected,
              "bank write changed under backpressure");
      end
      @(negedge clock);
      io_bankWrite_io_req_ready = 1;
      @(posedge clock);
      check(io_bankWrite_io_req_valid, "missing bank write handshake");
      ack_ref_write(model, io_bankWrite_io_req_bits_addr * 16, io_bankWrite_io_req_bits_data[63:0],
                    io_bankWrite_io_req_bits_data[127:64], 65535);
      @(negedge clock);
      io_bankWrite_io_req_ready = 0;
      no_completion(row == 3 ? 32 : 2);
      if (row == 3)
        `uvm_info("FINAL_ACK_DELAY",
                  "bankWrite request accepted; cmdResp.valid remained zero for 32 clocks without bankWrite ACK",
                  UVM_LOW)
      @(negedge clock);
      io_bankWrite_io_resp_bits_ok = 1;
      io_bankWrite_io_resp_valid   = 1;
      do @(posedge clock); while (!io_bankWrite_io_resp_ready);
      @(negedge clock);
      io_bankWrite_io_resp_valid = 0;
    end
    complete_command();
    for (int row = 0; row < 4; row++) begin
      expected = row_data(row);
      check(ack_ref_check(model, row * 16, expected[63:0], expected[127:64]) == 1,
            "bank golden read after completion failed");
    end
    // A failed bank ACK preserves the already-written prefix and drains all DMA beats.
    begin_load('h2000, 4);
    read_beat(0, 0);
    bank_ack(0, 1);
    read_beat(1, 0);
    bank_ack(1, 0);
    read_beat(2, 0, 4, 'hdead);
    no_bank_write(8);
    no_completion(16);
    read_beat(3, 1);
    no_bank_write(4);
    complete_command(5, 16);
    begin_load('h2100, 1);
    read_beat(0, 1);
    bank_ack(0, 1);
    complete_command();
    // Failed DMA data must never reach a bank; subsequent errors cannot replace the first.
    begin_load('h3000, 3);
    read_beat(0, 0, 3, 'h3030);
    no_bank_write(8);
    read_beat(1, 0);
    no_bank_write(8);
    no_completion(16);
    read_beat(2, 1, 2, 'h3040);
    no_bank_write(4);
    complete_command(3, 'h3030);
    begin_load('h3100, 1);
    read_beat(0, 1);
    bank_ack(0, 1);
    complete_command();
    begin_load('h8000, 2, 2, 3);
    read_beat(0, 1, 3, 'h8000);
    complete_command(3, 'h8000);
    // 2D footprint retains original rows instead of flattened DMA beat count.
    completions = 0;
    @(negedge clock);
    io_cmdReq_bits_cmd_mem_addr = 'ha000;
    io_cmdReq_bits_cmd_iter = 2;
    io_cmdReq_bits_cmd_is_mvin_2d = 1;
    io_cmdReq_bits_cmd_special = (64'd1 << 59) | (64'd1 << 36) | (64'd4 << 43);
    io_cmdReq_valid = 1;
    do @(posedge clock); while (!io_cmdReq_ready);
    @(negedge clock);
    io_cmdReq_valid = 0;
    wait (io_dmaReq_valid);
    check(
        io_footprint_valid && io_footprint_rows==2 && io_footprint_columns==2 &&
      io_footprint_spanBytes==16 && io_footprint_columnStride==8 && io_footprint_rowStride==32 &&
      io_footprint_fault_error==0 && io_dmaReq_bits_len==64,
        "2D padded footprint wrong");
    @(negedge clock);
    io_dmaReq_ready = 1;
    @(posedge clock);
    @(negedge clock);
    io_dmaReq_ready = 0;
    for (int row = 0; row < 4; row++) begin
      read_beat(row, row == 3);
      bank_ack(row, 1);
    end
    complete_command();
    @(negedge clock);
    io_cmdReq_bits_cmd_is_mvin_2d = 0;
    begin_load('ha100, 1);
    read_beat(0, 1);
    bank_ack(0, 1);
    complete_command();
    // MMIO explicitly uses one group despite an unrelated query value of two.
    completions = 0;
    @(negedge clock);
    io_query_group_count = 2;
    io_cmdReq_bits_cmd_mem_addr = 'hb000;
    io_cmdReq_bits_cmd_iter = 2;
    io_cmdReq_bits_cmd_is_mvin_mmio = 1;
    io_cmdReq_bits_cmd_special = (64'd16 << 56) | (64'h100 << 39);
    io_cmdReq_valid = 1;
    do @(posedge clock); while (!io_cmdReq_ready);
    @(negedge clock);
    io_cmdReq_valid = 0;
    wait (io_dmaReq_valid);
    check(
        io_footprint_valid && io_footprint_rows==2 && io_footprint_columns==1 &&
      io_footprint_spanBytes==16 && io_footprint_columnStride==0 && io_footprint_rowStride==16 &&
      io_footprint_fault_error==0 && io_dmaReq_bits_groups==1,
        "MMIO footprint must use one group");
    @(negedge clock);
    io_dmaReq_ready = 1;
    @(posedge clock);
    @(negedge clock);
    io_dmaReq_ready = 0;
    read_beat(0, 1, 3, 'hb000);
    complete_command(3, 'hb000);
    @(negedge clock);
    io_cmdReq_bits_cmd_is_mvin_mmio = 0;
    begin_load('hb100, 1);
    read_beat(0, 1);
    bank_ack(0, 1);
    complete_command();
    ack_ref_destroy(model);
    `uvm_info(
        "ACK_GATE",
        "Late ACK, typed DMA/bank fault, drain, stable completion and successful followup checked",
        UVM_LOW)
    ctl.done = 1;
  end
endmodule
