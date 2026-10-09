module memory_system_tb;
  import uvm_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  ip_control_if ctl (clock);
  `include "memory_system_signals.svh"
Memory dut (
      `include "memory_system_ports.svh"
  );
  bit line_valid[2], line_write[2], line_ready_out[2];
  bit [ 11:0] line_id  [2];
  bit [ 43:0] line_addr[2];
  bit [ 63:0] line_mask[2];
  bit [511:0] line_data[2];
  logic line_ready[2], reply_valid[2], reply_error[2];
  logic [ 11:0] reply_id  [2];
  logic [511:0] reply_data[2];
  assign io_lineRequest_0_valid = line_valid[0];
  assign io_lineRequest_0_bits_write = line_write[0];
  assign io_lineRequest_0_bits_id = line_id[0];
  assign io_lineRequest_0_bits_addr = line_addr[0];
  assign io_lineRequest_0_bits_data = line_data[0];
  assign io_lineRequest_0_bits_mask = line_mask[0];
  assign line_ready[0] = io_lineRequest_0_ready;
  assign io_lineResponse_0_ready = line_ready_out[0];
  assign reply_valid[0] = io_lineResponse_0_valid;
  assign reply_id[0] = io_lineResponse_0_bits_id;
  assign reply_data[0] = io_lineResponse_0_bits_data;
  assign reply_error[0] = io_lineResponse_0_bits_error;
  assign io_lineRequest_1_valid = line_valid[1];
  assign io_lineRequest_1_bits_write = line_write[1];
  assign io_lineRequest_1_bits_id = line_id[1];
  assign io_lineRequest_1_bits_addr = line_addr[1];
  assign io_lineRequest_1_bits_data = line_data[1];
  assign io_lineRequest_1_bits_mask = line_mask[1];
  assign line_ready[1] = io_lineRequest_1_ready;
  assign io_lineResponse_1_ready = line_ready_out[1];
  assign reply_valid[1] = io_lineResponse_1_valid;
  assign reply_id[1] = io_lineResponse_1_bits_id;
  assign reply_data[1] = io_lineResponse_1_bits_data;
  assign reply_error[1] = io_lineResponse_1_bits_error;
  `include "memory_system_clocking.svh"
  typedef struct {
    bit [63:0] addr;
    int beat;
  } read_t;
  typedef struct {
    bit [63:0] addr;
    bit error;
  } write_t;
  read_t reads[int];
  write_t writes[int];
  bit [511:0] memory[bit [63:0]];
  int aw_ids[$];
  bit [63:0] aw_addr[$];
  bit [511:0] w_data[$];
  bit [63:0] w_mask[$];
  bit [511:0] assembly;
  bit [63:0] assembly_mask;
  int beat = 0, cycles = 0, read_bursts = 0, write_bursts = 0, checks = 0;
  int dma_checks = 0;
  bit hold_b = 0, read_error = 0, write_error = 0;
  int r_id = 0, b_id = 0;
  bit r_active = 0, b_active = 0;
  function automatic void ck(bit ok, string message);
    if (!ok) `uvm_fatal("MEMORY_SYSTEM", message)
  endfunction
  task automatic axi_service();
    forever begin
      @(sample);
      if (!sample.reset) begin
        cycles++;
        if (sample.io_axi_aw_valid && sample.io_axi_aw_ready) begin
          ck(
              sample.io_axi_aw_bits_len==3&&sample.io_axi_aw_bits_size==4&&sample.io_axi_aw_bits_burst==1&&!sample.io_axi_aw_bits_lock,
              "AXI AW contract");
          aw_ids.push_back(sample.io_axi_aw_bits_id);
          aw_addr.push_back(sample.io_axi_aw_bits_addr);
        end
        if (sample.io_axi_w_valid && sample.io_axi_w_ready) begin
          ck(sample.io_axi_w_bits_last == (beat == 3), "AXI WLAST");
          assembly[128*beat+:128] = sample.io_axi_w_bits_data;
          assembly_mask[16*beat+:16] = sample.io_axi_w_bits_strb;
          if (beat == 3) begin
            w_data.push_back(assembly);
            w_mask.push_back(assembly_mask);
            beat = 0;
          end else beat++;
        end
        if (aw_ids.size() && w_data.size()) begin
          int id = aw_ids.pop_front();
          bit [63:0] addr = aw_addr.pop_front();
          bit [511:0] data = w_data.pop_front();
          bit [63:0] mask = w_mask.pop_front();
          ck(!writes.exists(id), "AXI active write ID reused");
          if (!write_error)
            for (int j = 0; j < 64; j++) if (mask[j]) memory[addr][j*8+:8] = data[j*8+:8];
          writes[id] = '{addr, write_error};
          write_bursts++;
        end
        if (sample.io_axi_ar_valid && sample.io_axi_ar_ready) begin
          int id = sample.io_axi_ar_bits_id;
          bit [63:0] addr = sample.io_axi_ar_bits_addr;
          ck(
              sample.io_axi_ar_bits_len==3&&sample.io_axi_ar_bits_size==4&&sample.io_axi_ar_bits_burst==1&&!sample.io_axi_ar_bits_lock,
              "AXI AR contract");
          ck(!reads.exists(id), "AXI active read ID reused");
          ck(memory.exists(addr), "Read of uninitialized external memory");
          reads[id] = '{addr, 0};
          read_bursts++;
        end
        if (sample.io_axi_r_valid && sample.io_axi_r_ready) begin
          if (reads[r_id].beat == 3) reads.delete(r_id);
          else reads[r_id].beat++;
          r_active = 0;
        end
        if (sample.io_axi_b_valid && sample.io_axi_b_ready) begin
          writes.delete(b_id);
          b_active = 0;
        end
      end
      @(negedge clock);
      io_axi_aw_ready = !reset && cycles % 5 != 0;
      io_axi_w_ready  = !reset && cycles % 7 >= 2;
      io_axi_ar_ready = !reset && cycles % 4 != 0;
      if (!r_active)
        foreach (reads[id])
        if (!r_active) begin
          r_active = 1;
          r_id = id;
          io_axi_r_bits_id = id;
          io_axi_r_bits_data = memory[reads[id].addr][128*reads[id].beat+:128];
          io_axi_r_bits_last = reads[id].beat == 3;
          io_axi_r_bits_resp = read_error && reads[id].beat == 1 ? 2 : 0;
        end
      if (!b_active && !hold_b)
        foreach (writes[id])
        if (!b_active) begin
          b_active = 1;
          b_id = id;
          io_axi_b_bits_id = id;
          io_axi_b_bits_resp = writes[id].error ? 2 : 0;
        end
      io_axi_r_valid = !reset && r_active;
      io_axi_b_valid = !reset && b_active;
    end
  endtask
  task automatic issue_line(int agent, bit write, bit [63:0] addr, bit [63:0] data, int id);
    @(negedge clock);
    line_valid[agent] = 1;
    line_write[agent] = write;
    line_addr[agent] = addr;
    line_data[agent] = '0;
    line_data[agent][63:0] = data;
    line_mask[agent] = 'hff;
    line_id[agent] = id;
    do @(sample); while (!sample.line_ready[agent]);
    @(negedge clock);
    line_valid[agent] = 0;
  endtask
  task automatic receive_line(int agent, int id, bit [511:0] data, bit write = 0);
    do @(sample); while (!sample.reply_valid[agent]);
    repeat (4) begin
      ck(sample.reply_valid[agent] && sample.reply_id[agent] == id && !sample.reply_error[agent],
         "Line owner/error or backpressure stability");
      if (!write) ck(sample.reply_data[agent] == data, "DDR line readback data");
      @(sample);
    end
    @(negedge clock);
    line_ready_out[agent] = 1;
    do @(sample); while (!sample.reply_valid[agent]);
    checks++;
    @(negedge clock);
    line_ready_out[agent] = 0;
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 1ms);
    run_test("protocol_test");
  end
  initial begin
    io_dma_0_aw_valid = 0;
    io_dma_0_aw_bits_id = 6;
    io_dma_0_aw_bits_addr = 'h80000080;
    io_dma_0_aw_bits_len = 3;
    io_dma_0_aw_bits_size = 4;
    io_dma_0_aw_bits_burst = 1;
    io_dma_0_aw_bits_lock = 0;
    io_dma_0_aw_bits_cache = 0;
    io_dma_0_aw_bits_prot = 0;
    io_dma_0_aw_bits_qos = 0;
    io_dma_0_aw_bits_region = 0;
    io_dma_0_w_valid = 0;
    io_dma_0_w_bits_data = 0;
    io_dma_0_w_bits_strb = '1;
    io_dma_0_w_bits_last = 0;
    io_dma_0_b_ready = 0;
    io_dma_0_ar_valid = 0;
    io_dma_0_ar_bits_id = 6;
    io_dma_0_ar_bits_addr = 'h80000080;
    io_dma_0_ar_bits_len = 3;
    io_dma_0_ar_bits_size = 4;
    io_dma_0_ar_bits_burst = 1;
    io_dma_0_ar_bits_lock = 0;
    io_dma_0_ar_bits_cache = 0;
    io_dma_0_ar_bits_prot = 0;
    io_dma_0_ar_bits_qos = 0;
    io_dma_0_ar_bits_region = 0;
    io_dma_0_r_ready = 0;
    for (int i = 0; i < 2; i++) begin
      line_valid[i] = 0;
      line_write[i] = 0;
      line_addr[i] = 0;
      line_data[i] = 0;
      line_mask[i] = 0;
      line_id[i] = 0;
      line_ready_out[i] = 0;
    end
    io_axi_aw_ready = 0;
    io_axi_ar_ready = 0;
    io_axi_w_ready = 0;
    io_axi_r_valid = 0;
    io_axi_b_valid = 0;
    io_axi_r_bits_id = 0;
    io_axi_r_bits_data = 0;
    io_axi_r_bits_last = 0;
    io_axi_r_bits_resp = 0;
    io_axi_b_bits_id = 0;
    io_axi_b_bits_resp = 0;
    memory['h80000000] = '0;
    memory['h80000040] = '0;
    wait (ctl.start);
    repeat (5) @(negedge clock);
    reset = 0;
    fork
      axi_service();
    join_none
    // A retained NPU write response must not block independent coherence backing reads.
    hold_b = 1;
    @(negedge clock);
    io_dma_0_aw_valid = 1;
    do @(sample); while (!sample.io_dma_0_aw_ready);
    @(negedge clock);
    io_dma_0_aw_valid = 0;
    for (int i = 0; i < 4; i++) begin
      io_dma_0_w_valid = 1;
      io_dma_0_w_bits_data = 128'('h100 + i);
      io_dma_0_w_bits_last = i == 3;
      do @(sample); while (!sample.io_dma_0_w_ready);
      @(negedge clock);
    end
    io_dma_0_w_valid = 0;
    wait (writes.num() != 0);
    issue_line(0, 0, 'h80000000, 0, 19);
    receive_line(0, 19, 0);
    ck(io_outstanding == 1, "NPU write completed before its final AXI B");
    @(negedge clock);
    hold_b = 0;
    io_dma_0_b_ready = 1;
    do @(sample); while (!sample.io_dma_0_b_valid);
    ck(sample.io_dma_0_b_bits_id == 6 && sample.io_dma_0_b_bits_resp == 0,
       "NPU write response owner or error");
    @(negedge clock);
    io_dma_0_b_ready  = 0;
    io_dma_0_ar_valid = 1;
    do @(sample); while (!sample.io_dma_0_ar_ready);
    @(negedge clock);
    io_dma_0_ar_valid = 0;
    io_dma_0_r_ready  = 1;
    for (int i = 0; i < 4; i++) begin
      do @(sample); while (!sample.io_dma_0_r_valid);
      ck(
          sample.io_dma_0_r_bits_id == 6 && sample.io_dma_0_r_bits_resp == 0 &&
         sample.io_dma_0_r_bits_data == 128'('h100 + i) && sample.io_dma_0_r_bits_last == (i == 3),
          "NPU readback data, owner, error or RLAST");
      dma_checks++;
    end
    @(negedge clock);
    io_dma_0_r_ready = 0;
    fork
      begin
        issue_line(0, 1, 'h80000000, 23, 20);
        receive_line(0, 20, 0, 1);
      end
      begin
        issue_line(1, 1, 'h80000040, 31, 20);
        receive_line(1, 20, 0, 1);
      end
    join
    fork
      begin
        issue_line(0, 0, 'h80000000, 0, 21);
        receive_line(0, 21, 23);
      end
      begin
        issue_line(1, 0, 'h80000040, 0, 21);
        receive_line(1, 21, 31);
      end
    join
    // Neither client observes write completion before the final AXI B.
    hold_b = 1;
    issue_line(0, 1, 'h80000000, 41, 22);
    wait (writes.num() != 0);
    repeat (12) begin
      @(sample);
      ck(!sample.reply_valid[0] && sample.io_outstanding != 0,
         "Backing write completed before final AXI B");
    end
    @(negedge clock);
    hold_b = 0;
    receive_line(0, 22, 0, 1);
    // Transport errors return to the originating coherence client.
    read_error = 1;
    issue_line(1, 0, 'h80000040, 0, 23);
    do @(sample); while (!sample.reply_valid[1]);
    ck(sample.reply_id[1] == 23 && sample.reply_error[1], "DDR R error owner");
    @(negedge clock);
    line_ready_out[1] = 1;
    do @(sample); while (!sample.reply_valid[1]);
    checks++;
    @(negedge clock);
    line_ready_out[1] = 0;
    read_error = 0;
    write_error = 1;
    issue_line(0, 1, 'h80000000, 99, 24);
    do @(sample); while (!sample.reply_valid[0]);
    ck(sample.reply_id[0] == 24 && sample.reply_error[0], "DDR B error owner");
    @(negedge clock);
    line_ready_out[0] = 1;
    do @(sample); while (!sample.reply_valid[0]);
    checks++;
    @(negedge clock);
    line_ready_out[0] = 0;
    write_error = 0;
    issue_line(0, 0, 'h80000000, 0, 25);
    receive_line(0, 25, 41);
    repeat (8) @(sample);
    ck(
        !reads.num() && !writes.num() && !aw_ids.size() && !aw_addr.size() &&
       !w_data.size() && !w_mask.size() && !r_active && !b_active && sample.io_outstanding == 0,
        "DDR transport did not drain");
    for (int i = 0; i < 2; i++) ck(!sample.reply_valid[i], "Unmatched backing response");
    ck(checks == 9 && dma_checks == 4, "Missing checked backing/DMA transactions");
    `uvm_info("MEMORY_SYSTEM_PASS", $sformatf(
              "backing checked=%0d DMA beats=%0d AXI reads/writes=%0d/%0d; owners, backpressure, final B, R/B errors and drain checked",
              checks,
              dma_checks,
              read_bursts,
              write_bursts
              ), UVM_LOW)
    ctl.done = 1;
  end
endmodule
