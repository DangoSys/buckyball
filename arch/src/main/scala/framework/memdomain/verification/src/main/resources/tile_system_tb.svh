`include "tile_system_config.svh"
import uvm_pkg::*;
import ip_control_test_pkg::*;
`include "uvm_macros.svh"
import "DPI-C" function longint unsigned tile_memory_init(
  input string imagePath,
  input string expectedPath
);
import "DPI-C" function longint unsigned tile_peer_read64(input longint unsigned address);
import "DPI-C" function void tile_peer_write128(
  input longint unsigned address,
  input longint unsigned lo,
  input longint unsigned hi,
  input int unsigned mask
);
import "DPI-C" function void tile_peer_finish();
import "DPI-C" function void dpi_bdb_set_clk(input longint unsigned cycle);
logic clock = 0, reset = 1;
always #5 clock = ~clock;
ip_control_if ctl (clock);
`include "tile_system_signals.svh"
TileSystem dut (
    `include "tile_system_ports.svh"
);
logic device_valid[5], device_ready[5], device_write[5], device_normal[5];
logic [63:0] device_addr[5], device_data[5];
logic [5:0] device_tag[5];
logic [2:0] device_size[5];
logic [3:0] device_atomic[5];
logic dev_response_valid[5], dev_response_ready[5];
logic [ 5:0] dev_response_tag [5];
logic [63:0] dev_response_data[5];
logic retired[5], trapped[5], failure[5];
logic [63:0] retired_pc[5], trap_pc[5], trap_cause[5], trap_value[5];
assign device_valid[0] = io_deviceRequest_0_valid;
assign device_addr[0] = io_deviceRequest_0_bits_addr;
assign device_data[0] = io_deviceRequest_0_bits_data;
assign device_tag[0] = io_deviceRequest_0_bits_tag;
assign device_size[0] = io_deviceRequest_0_bits_size;
assign device_atomic[0] = io_deviceRequest_0_bits_atomic;
assign device_write[0] = io_deviceRequest_0_bits_write;
assign device_normal[0] = io_deviceRequest_0_bits_normal;
assign io_deviceRequest_0_ready = device_ready[0];
assign dev_response_ready[0] = io_deviceResponse_0_ready;
assign io_deviceResponse_0_valid = dev_response_valid[0];
assign io_deviceResponse_0_bits_tag = dev_response_tag[0];
assign io_deviceResponse_0_bits_data = dev_response_data[0];
assign io_deviceResponse_0_bits_error = 0;
assign retired[0] = io_retired_0;
assign retired_pc[0] = io_retiredPc_0;
assign trapped[0] = io_trapped_0;
assign trap_pc[0] = io_trapPc_0;
assign trap_cause[0] = io_trapCause_0;
assign trap_value[0] = io_trapValue_0;
assign failure[0] = io_failure_0_valid;
assign device_valid[1] = io_deviceRequest_1_valid;
assign device_addr[1] = io_deviceRequest_1_bits_addr;
assign device_data[1] = io_deviceRequest_1_bits_data;
assign device_tag[1] = io_deviceRequest_1_bits_tag;
assign device_size[1] = io_deviceRequest_1_bits_size;
assign device_atomic[1] = io_deviceRequest_1_bits_atomic;
assign device_write[1] = io_deviceRequest_1_bits_write;
assign device_normal[1] = io_deviceRequest_1_bits_normal;
assign io_deviceRequest_1_ready = device_ready[1];
assign dev_response_ready[1] = io_deviceResponse_1_ready;
assign io_deviceResponse_1_valid = dev_response_valid[1];
assign io_deviceResponse_1_bits_tag = dev_response_tag[1];
assign io_deviceResponse_1_bits_data = dev_response_data[1];
assign io_deviceResponse_1_bits_error = 0;
assign retired[1] = io_retired_1;
assign retired_pc[1] = io_retiredPc_1;
assign trapped[1] = io_trapped_1;
assign trap_pc[1] = io_trapPc_1;
assign trap_cause[1] = io_trapCause_1;
assign trap_value[1] = io_trapValue_1;
assign failure[1] = io_failure_1_valid;
assign device_valid[2] = io_deviceRequest_2_valid;
assign device_addr[2] = io_deviceRequest_2_bits_addr;
assign device_data[2] = io_deviceRequest_2_bits_data;
assign device_tag[2] = io_deviceRequest_2_bits_tag;
assign device_size[2] = io_deviceRequest_2_bits_size;
assign device_atomic[2] = io_deviceRequest_2_bits_atomic;
assign device_write[2] = io_deviceRequest_2_bits_write;
assign device_normal[2] = io_deviceRequest_2_bits_normal;
assign io_deviceRequest_2_ready = device_ready[2];
assign dev_response_ready[2] = io_deviceResponse_2_ready;
assign io_deviceResponse_2_valid = dev_response_valid[2];
assign io_deviceResponse_2_bits_tag = dev_response_tag[2];
assign io_deviceResponse_2_bits_data = dev_response_data[2];
assign io_deviceResponse_2_bits_error = 0;
assign retired[2] = io_retired_2;
assign retired_pc[2] = io_retiredPc_2;
assign trapped[2] = io_trapped_2;
assign trap_pc[2] = io_trapPc_2;
assign trap_cause[2] = io_trapCause_2;
assign trap_value[2] = io_trapValue_2;
assign failure[2] = io_failure_2_valid;
assign device_valid[3] = io_deviceRequest_3_valid;
assign device_addr[3] = io_deviceRequest_3_bits_addr;
assign device_data[3] = io_deviceRequest_3_bits_data;
assign device_tag[3] = io_deviceRequest_3_bits_tag;
assign device_size[3] = io_deviceRequest_3_bits_size;
assign device_atomic[3] = io_deviceRequest_3_bits_atomic;
assign device_write[3] = io_deviceRequest_3_bits_write;
assign device_normal[3] = io_deviceRequest_3_bits_normal;
assign io_deviceRequest_3_ready = device_ready[3];
assign dev_response_ready[3] = io_deviceResponse_3_ready;
assign io_deviceResponse_3_valid = dev_response_valid[3];
assign io_deviceResponse_3_bits_tag = dev_response_tag[3];
assign io_deviceResponse_3_bits_data = dev_response_data[3];
assign io_deviceResponse_3_bits_error = 0;
assign retired[3] = io_retired_3;
assign retired_pc[3] = io_retiredPc_3;
assign trapped[3] = io_trapped_3;
assign trap_pc[3] = io_trapPc_3;
assign trap_cause[3] = io_trapCause_3;
assign trap_value[3] = io_trapValue_3;
assign failure[3] = io_failure_3_valid;
assign device_valid[4] = io_deviceRequest_4_valid;
assign device_addr[4] = io_deviceRequest_4_bits_addr;
assign device_data[4] = io_deviceRequest_4_bits_data;
assign device_tag[4] = io_deviceRequest_4_bits_tag;
assign device_size[4] = io_deviceRequest_4_bits_size;
assign device_atomic[4] = io_deviceRequest_4_bits_atomic;
assign device_write[4] = io_deviceRequest_4_bits_write;
assign device_normal[4] = io_deviceRequest_4_bits_normal;
assign io_deviceRequest_4_ready = device_ready[4];
assign dev_response_ready[4] = io_deviceResponse_4_ready;
assign io_deviceResponse_4_valid = dev_response_valid[4];
assign io_deviceResponse_4_bits_tag = dev_response_tag[4];
assign io_deviceResponse_4_bits_data = dev_response_data[4];
assign io_deviceResponse_4_bits_error = 0;
assign retired[4] = io_retired_4;
assign retired_pc[4] = io_retiredPc_4;
assign trapped[4] = io_trapped_4;
assign trap_pc[4] = io_trapPc_4;
assign trap_cause[4] = io_trapCause_4;
assign trap_value[4] = io_trapValue_4;
assign failure[4] = io_failure_4_valid;
clocking sample @(posedge clock);
  default input #1step;
  input reset;
  input device_valid,device_ready,device_write,device_normal,device_addr,device_data,device_tag,device_size,device_atomic;
  input dev_response_valid,dev_response_ready,retired,retired_pc,trapped,trap_pc,trap_cause,trap_value,failure;
  input io_memoryOutstanding;
  input io_axi_aw_valid,io_axi_aw_ready,io_axi_aw_bits_id,io_axi_aw_bits_addr,io_axi_aw_bits_len,io_axi_aw_bits_size,io_axi_aw_bits_burst,io_axi_aw_bits_lock;
  input io_axi_w_valid, io_axi_w_ready, io_axi_w_bits_data, io_axi_w_bits_strb, io_axi_w_bits_last;
  input io_axi_ar_valid,io_axi_ar_ready,io_axi_ar_bits_id,io_axi_ar_bits_addr,io_axi_ar_bits_len,io_axi_ar_bits_size,io_axi_ar_bits_burst,io_axi_ar_bits_lock;
  input io_axi_r_valid, io_axi_r_ready, io_axi_b_valid, io_axi_b_ready;
endclocking
int cycles = 0, retire_count[5], device_requests = 0, device_responses = 0;
bit exit_seen = 0, exit_ack = 0;
int exit_status = 0;
bit device_pending[5], device_exit[5];
int device_due[5];
bit [5:0] pending_tag[5];
bit [63:0] pending_data[5];
function automatic void ck(bit ok, string message);
  if (!ok) `uvm_fatal("TILE_SYSTEM", message)
endfunction
`include "tile_axi_peer.svh"
task automatic device_service();
  forever begin
    @(sample);
    if (!sample.reset)
      for (int i = 0; i < 5; i++) begin
        if (sample.device_valid[i] && sample.device_ready[i]) begin
          ck(!device_pending[i] && !sample.device_normal[i] && sample.device_atomic[i] == 0,
             "Illegal actual Tile Device request/owner");
          device_pending[i] = 1;
          pending_tag[i] = sample.device_tag[i];
          pending_data[i] = 0;
          device_due[i] = cycles + 4;
          device_exit[i] = 0;
          device_requests++;
          case (sample.device_addr[i])
            'h60000000: begin
              ck(i == 0 && sample.device_write[i] && sample.device_size[i] == 2 && !exit_seen,
                 "Completion must be one controller 32-bit Device write");
              exit_seen = 1;
              exit_status = int'(sample.device_data[i][31:0]);
              device_exit[i] = 1;
              ck(exit_status == 0, $sformatf(
                 "CPU firmware exited status%0d pc%h", exit_status, sample.retired_pc[i]));
            end
            'h10000000: begin
              if (sample.device_write[i]) $write("%c", sample.device_data[i][7:0]);
            end
            'h10000005: begin
              ck(!sample.device_write[i], "UART status is read only");
              pending_data[i] = 'h60;
            end
            default:
            ck(0, $sformatf(
               "Unsupported actual Device address hart%0d addr%h", i, sample.device_addr[i]));
          endcase
        end
        if (sample.dev_response_valid[i] && sample.dev_response_ready[i]) begin
          ck(device_pending[i], "Device response without retained CPU owner");
          if (device_exit[i]) exit_ack = 1;
          device_pending[i] = 0;
          device_responses++;
        end
      end
    @(negedge clock);
    for (int i = 0; i < 5; i++) begin
      device_ready[i] = !reset && !device_pending[i] && cycles % 5 != 0;
      dev_response_valid[i] = !reset && device_pending[i] && device_due[i] <= cycles;
      dev_response_tag[i] = pending_tag[i];
      dev_response_data[i] = pending_data[i];
    end
  end
endtask
task automatic monitor_cpu();
  forever begin
    @(sample);
    if (!sample.reset) begin
      cycles++;
      dpi_bdb_set_clk(cycles);
      for (int i = 0; i < 5; i++) begin
        ck(!sample.failure[i], $sformatf("Actual NPU DMA failure hart%0d", i));
        if (sample.retired[i]) retire_count[i]++;
        if (sample.trapped[i]) begin
          ck(i != 0 && sample.trap_cause[i] == 8, $sformatf(
             "Unexpected CPU trap hart%0d pc%h cause%h value%h",
             i,
             sample.trap_pc[i],
             sample.trap_cause[i],
             sample.trap_value[i]
             ));
        end
      end
    end
  end
endtask
initial begin
  uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
  uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 1ms);
  run_test("protocol_test");
end
initial begin
  longint unsigned entry;
  ck(`TILE_CPU_COUNT == 5, "Tile gate requires actual Goban five-CPU profile");
  for (int i = 0; i < 5; i++) begin
    device_ready[i] = 0;
    dev_response_valid[i] = 0;
    dev_response_tag[i] = 0;
    dev_response_data[i] = 0;
    device_pending[i] = 0;
    device_exit[i] = 0;
    device_due[i] = 0;
    pending_tag[i] = 0;
    pending_data[i] = 0;
    retire_count[i] = 0;
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
  io_interrupts_0_external = 0;
  io_interrupts_0_software = 0;
  io_interrupts_0_timer = 0;
  io_interrupts_1_external = 0;
  io_interrupts_1_software = 0;
  io_interrupts_1_timer = 0;
  io_interrupts_2_external = 0;
  io_interrupts_2_software = 0;
  io_interrupts_2_timer = 0;
  io_interrupts_3_external = 0;
  io_interrupts_3_software = 0;
  io_interrupts_3_timer = 0;
  io_interrupts_4_external = 0;
  io_interrupts_4_software = 0;
  io_interrupts_4_timer = 0;
  entry = tile_memory_init(`TILE_IMAGE_PATH, `TILE_EXPECT_PATH);
  io_resetVector_0 = entry;
  io_resetVector_1 = entry;
  io_resetVector_2 = entry;
  io_resetVector_3 = entry;
  io_resetVector_4 = entry;
  wait (ctl.start);
  repeat (5) @(negedge clock);
  reset = 0;
  fork
    axi_service();
    device_service();
    monitor_cpu();
  join_none
  wait (exit_ack);
  do
  @(sample);
  while (reads.num() || writes.num() || aw_ids.size() || aw_addresses.size() || w_lines.size() ||
         w_masks.size() || w_beat || r_active || b_active || sample.io_memoryOutstanding != 0);
  ck(device_requests == device_responses, "Device completions not drained");
  for (int i = 0; i < 5; i++)
  ck(retire_count[i] > 0, $sformatf("Configured CPU%0d never executed", i));
  ck(axi_reads > 0 && axi_writes > 0 && final_b_waits > 0,
     "Firmware did not exercise actual DDR read/write/finalB");
  tile_peer_finish();
  `uvm_info("TILE_SYSTEM_PASS", $sformatf(
            "actual five CPU firmware cycles%0d DDR read/write%0d/%0d finalBwait%0d Device%0d/%0d status%0d; no RoCC BFM",
            cycles,
            axi_reads,
            axi_writes,
            final_b_waits,
            device_requests,
            device_responses,
            exit_status
            ), UVM_LOW)
  ctl.done = 1;
end
