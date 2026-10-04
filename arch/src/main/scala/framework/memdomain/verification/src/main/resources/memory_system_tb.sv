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
  bit
      req_valid[2],
      req_normal[2],
      req_write[2],
      rsp_ready[2],
      dev_ready[2],
      dev_valid[2],
      dev_error[2];
  logic req_ready[2], rsp_valid[2], rsp_error[2], device_valid[2], device_resp_ready[2];
  bit [63:0] req_addr[2], req_data[2], dev_data[2];
  bit [5:0] req_tag[2], dev_tag[2];
  bit [2:0] req_size  [2];
  bit [3:0] req_atomic[2];
  logic [5:0] rsp_tag[2], device_tag[2];
  logic [63:0] rsp_data[2];
  bit line_valid[2], line_write[2], line_ready_out[2];
  bit [ 11:0] line_id  [2];
  bit [ 43:0] line_addr[2];
  bit [ 63:0] line_mask[2];
  bit [511:0] line_data[2];
  logic line_ready[2], reply_valid[2], reply_error[2];
  logic [ 11:0] reply_id  [2];
  logic [511:0] reply_data[2];
  assign io_cpuRequest_0_valid = req_valid[0];
  assign io_cpuRequest_0_bits_normal = req_normal[0];
  assign io_cpuRequest_0_bits_write = req_write[0];
  assign io_cpuRequest_0_bits_addr = req_addr[0];
  assign io_cpuRequest_0_bits_data = req_data[0];
  assign io_cpuRequest_0_bits_tag = req_tag[0];
  assign io_cpuRequest_0_bits_size = req_size[0];
  assign io_cpuRequest_0_bits_atomic = req_atomic[0];
  assign req_ready[0] = io_cpuRequest_0_ready;
  assign rsp_valid[0] = io_cpuResponse_0_valid;
  assign rsp_tag[0] = io_cpuResponse_0_bits_tag;
  assign rsp_data[0] = io_cpuResponse_0_bits_data;
  assign rsp_error[0] = io_cpuResponse_0_bits_error;
  assign io_cpuResponse_0_ready = rsp_ready[0];
  assign io_deviceRequest_0_ready = dev_ready[0];
  assign device_valid[0] = io_deviceRequest_0_valid;
  assign device_tag[0] = io_deviceRequest_0_bits_tag;
  assign device_resp_ready[0] = io_deviceResponse_0_ready;
  assign io_deviceResponse_0_valid = dev_valid[0];
  assign io_deviceResponse_0_bits_tag = dev_tag[0];
  assign io_deviceResponse_0_bits_data = dev_data[0];
  assign io_deviceResponse_0_bits_error = dev_error[0];
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
  assign io_cpuRequest_1_valid = req_valid[1];
  assign io_cpuRequest_1_bits_normal = req_normal[1];
  assign io_cpuRequest_1_bits_write = req_write[1];
  assign io_cpuRequest_1_bits_addr = req_addr[1];
  assign io_cpuRequest_1_bits_data = req_data[1];
  assign io_cpuRequest_1_bits_tag = req_tag[1];
  assign io_cpuRequest_1_bits_size = req_size[1];
  assign io_cpuRequest_1_bits_atomic = req_atomic[1];
  assign req_ready[1] = io_cpuRequest_1_ready;
  assign rsp_valid[1] = io_cpuResponse_1_valid;
  assign rsp_tag[1] = io_cpuResponse_1_bits_tag;
  assign rsp_data[1] = io_cpuResponse_1_bits_data;
  assign rsp_error[1] = io_cpuResponse_1_bits_error;
  assign io_cpuResponse_1_ready = rsp_ready[1];
  assign io_deviceRequest_1_ready = dev_ready[1];
  assign device_valid[1] = io_deviceRequest_1_valid;
  assign device_tag[1] = io_deviceRequest_1_bits_tag;
  assign device_resp_ready[1] = io_deviceResponse_1_ready;
  assign io_deviceResponse_1_valid = dev_valid[1];
  assign io_deviceResponse_1_bits_tag = dev_tag[1];
  assign io_deviceResponse_1_bits_data = dev_data[1];
  assign io_deviceResponse_1_bits_error = dev_error[1];
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
  int beat = 0, cycles = 0, read_bursts = 0, write_bursts = 0, checks = 0, device_count = 0;
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
          ck(!reads.exists(id) && !writes.exists(id), "AXI active ID reused on write");
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
          ck(!reads.exists(id) && !writes.exists(id), "AXI active ID reused on read");
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
  task automatic issue_cpu(int cpu, bit normal, bit write, int atomic, bit [63:0] addr,
                           bit [63:0] data, int tag);
    @(negedge clock);
    req_normal[cpu] = normal;
    req_write[cpu] = write;
    req_atomic[cpu] = atomic;
    req_addr[cpu] = addr;
    req_data[cpu] = data;
    req_tag[cpu] = tag;
    req_size[cpu] = 3;
    req_valid[cpu] = 1;
    do @(sample); while (!sample.req_ready[cpu]);
    @(negedge clock);
    req_valid[cpu] = 0;
  endtask
  task automatic receive_cpu(int cpu, int tag, bit [63:0] data, bit error = 0);
    do @(sample); while (!sample.rsp_valid[cpu]);
    repeat (4) begin
      ck(
          sample.rsp_valid[cpu]&&sample.rsp_tag[cpu]==tag&&sample.rsp_data[cpu]==data&&sample.rsp_error[cpu]==error&&!sample.req_ready[cpu],
          "CPU owner/tag/data/error or backpressure stability");
      if (dev_valid[cpu])
        ck(!sample.device_resp_ready[cpu], "Device response ignored CPU backpressure");
      @(sample);
    end
    @(negedge clock);
    rsp_ready[cpu] = 1;
    do @(sample); while (!sample.rsp_valid[cpu]);
    checks++;
    @(negedge clock);
    rsp_ready[cpu] = 0;
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
  task automatic receive_line(int agent, int id, bit [63:0] data, bit write = 0);
    do @(sample); while (!sample.reply_valid[agent]);
    repeat (4) begin
      ck(sample.reply_valid[agent] && sample.reply_id[agent] == id && !sample.reply_error[agent],
         "Line owner/error or backpressure stability");
      if (!write)
        ck(sample.reply_data[agent][63:0] == data, "Line data after centralized CPU atomics");
      @(sample);
    end
    @(negedge clock);
    line_ready_out[agent] = 1;
    do @(sample); while (!sample.reply_valid[agent]);
    checks++;
    @(negedge clock);
    line_ready_out[agent] = 0;
  endtask
  task automatic device_transaction(int cpu, int tag, bit error, bit isolated = 1);
    dev_ready[cpu] = 0;
    fork
      issue_cpu(cpu, 0, 0, 0, 'h10000000 + cpu * 8, 0, tag);
      begin
        do @(sample); while (!sample.device_valid[cpu]);
        repeat (5) begin
          ck(sample.device_valid[cpu] && sample.device_tag[cpu] == tag && !sample.req_ready[cpu],
             "Device request backpressure owner/tag");
          if (isolated)
            ck(!sample.io_axi_ar_valid && !sample.io_axi_aw_valid, "Device request entered DDR");
          @(sample);
        end
        @(negedge clock);
        dev_ready[cpu] = 1;
      end
    join
    @(negedge clock);
    dev_valid[cpu] = 1;
    dev_tag[cpu]   = tag;
    dev_data[cpu]  = 'hd000 + tag;
    dev_error[cpu] = error;
    receive_cpu(cpu, tag, 'hd000 + tag, error);
    @(negedge clock);
    dev_valid[cpu] = 0;
    dev_ready[cpu] = 0;
    device_count++;
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 1ms);
    run_test("protocol_test");
  end
  initial begin
    foreach (req_valid[i]) begin
      req_valid[i] = 0;
      req_normal[i] = 0;
      req_write[i] = 0;
      req_addr[i] = 0;
      req_data[i] = 0;
      req_tag[i] = 0;
      req_size[i] = 3;
      req_atomic[i] = 0;
      rsp_ready[i] = 0;
      dev_ready[i] = 0;
      dev_valid[i] = 0;
      dev_data[i] = 0;
      dev_tag[i] = 0;
      dev_error[i] = 0;
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
    // Alternate retained device and normal ownership independently on each CPU.
    for (int cpu = 0; cpu < 2; cpu++) begin
      device_transaction(cpu, 1 + cpu, 0);
      issue_cpu(cpu, 1, 0, 0, 'h80000000, 0, 3 + cpu);
      receive_cpu(cpu, 3 + cpu, 0);
      device_transaction(cpu, 5 + cpu, 1);
    end
    fork
      device_transaction(1, 7, 0, 0);
      begin
        issue_cpu(0, 1, 0, 0, 'h80000000, 0, 8);
        receive_cpu(0, 8, 0);
      end
    join
    // Hold a real AMO write's AXI B; a backing line reader must not pass it.
    hold_b = 1;
    issue_cpu(0, 1, 0, 2, 'h80000000, 5, 9);
    wait (writes.num() != 0);
    fork
      issue_line(0, 0, 'h80000000, 0, 19);
      begin
        repeat (20) begin
          @(sample);
          ck(!sample.rsp_valid[0] && !sample.reply_valid[0] && !sample.line_ready[0],
             "AMO/line response escaped actual final AXI B");
        end
        @(negedge clock);
        hold_b = 0;
      end
    join
    fork
      receive_cpu(0, 9, 0);
      receive_line(0, 19, 5);
    join
    // LR reservation is invalidated by a Tile/NPU backing line write.
    issue_cpu(1, 1, 0, 10, 'h80000000, 0, 10);
    receive_cpu(1, 10, 5);
    issue_line(1, 1, 'h80000000, 23, 20);
    receive_line(1, 20, 0, 1);
    issue_cpu(1, 1, 0, 11, 'h80000000, 99, 11);
    receive_cpu(1, 11, 1);
    issue_cpu(0, 1, 0, 0, 'h80000000, 0, 12);
    receive_cpu(0, 12, 23);
    issue_cpu(0, 1, 0, 10, 'h80000000, 0, 13);
    receive_cpu(0, 13, 23);
    issue_cpu(0, 1, 0, 11, 'h80000000, 31, 14);
    receive_cpu(0, 14, 0);
    issue_line(0, 0, 'h80000000, 0, 21);
    receive_line(0, 21, 31);
    // Actual AXI failure responses reach the CPU owner.
    read_error = 1;
    issue_cpu(1, 1, 0, 0, 'h80000040, 0, 15);
    receive_cpu(1, 15, 0, 1);
    read_error  = 0;
    write_error = 1;
    issue_cpu(0, 1, 1, 0, 'h80000040, 77, 16);
    receive_cpu(0, 16, 0, 1);
    write_error = 0;
    issue_cpu(1, 1, 0, 0, 'h80000040, 0, 17);
    receive_cpu(1, 17, 0);
    repeat (8) @(sample);
    ck(
        !reads.num()&&!writes.num()&&!aw_ids.size()&&!aw_addr.size()&&!w_data.size()&&!w_mask.size()&&!r_active&&!b_active&&io_outstanding==0,
        "Memory/AXI did not drain");
    for (int i = 0; i < 2; i++)
    ck(!sample.rsp_valid[i] && !sample.reply_valid[i],
       "Unmatched CPU or line response after drain");
    ck(checks == 20 && device_count == 5 && read_bursts > 0 && write_bursts > 0,
       "Required checked transactions missing");
    `uvm_info("MEMORY_SYSTEM_PASS", $sformatf(
              "production Memory checked=%0d device=%0d AXI reads/writes=%0d/%0d; CPU/line ownership backpressure AMO real-finalB LR/SC line invalidation R/B errors drained",
              checks,
              device_count,
              read_bursts,
              write_bursts
              ), UVM_LOW)
    ctl.done = 1;
  end
endmodule
