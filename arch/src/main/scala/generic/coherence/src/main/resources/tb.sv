`include "coherence_config.svh"
module coherence_tb;
  import uvm_pkg::*;
  import coherence_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  coherence_control_if control (clock);
  stream_if #(`COH_REQ_WIDTH) source_if (
      clock,
      control.reset
  );
  stream_if #(`COH_RSP_WIDTH) rx_rsp (
      clock,
      control.reset
  );
  stream_if #(`COH_DAT_WIDTH) rx_dat (
      clock,
      control.reset
  );
  stream_if #(`COH_TXRSP_WIDTH) tx_rsp (
      clock,
      control.reset
  );
  stream_if #(`COH_TXDAT_WIDTH) sink_if (
      clock,
      control.reset
  );
  stream_if #(`COH_SNP_CHANNEL_WIDTH) tx_snp (
      clock,
      control.reset
  );
  stream_if #(`COH_MEMREQ_WIDTH) mem_req (
      clock,
      control.reset
  );
  stream_if #(`COH_MEMRESP_WIDTH) mem_resp (
      clock,
      control.reset
  );
  Coherence dut (
      .clock(clock),
      .reset(control.reset),
      .io_outstanding(control.outstanding),
      `include "coherence_ports.svh"
  );
  // Verify indirect payload reads against the value owned when the command entered the queue.
  wire [`COH_MSHRS-1:0][511:0] cache_payloads = `COH_CACHE_PAYLOADS;
  wire [`COH_MSHRS-1:0] cache_active = `COH_ACTIVE_SLOTS;
  logic [`COH_MSHRS-1:0] cache_queued = '0;
  logic [`COH_MSHRS-1:0][511:0] cache_snapshots;
  longint cache_enqueues = 0, cache_accepts = 0, cache_stalls = 0;
  longint cache_response_overlap = 0, cache_memory_overlap = 0;
  always @(posedge clock) begin
    if (control.reset) cache_queued <= '0;
    else begin
      if (dut.cacheQueue.io_enq_valid && dut.cacheQueue.io_enq_ready) cache_enqueues++;
      if (dut.cache.io_request_valid && dut.cache.io_request_ready) cache_accepts++;
      if (dut.cache.io_request_valid && !dut.cache.io_request_ready) cache_stalls++;
      if (dut.cacheQueue.io_enq_valid && dut.cacheQueue.io_enq_ready && dut.cache.io_response_valid)
        cache_response_overlap++;
      if (dut.cacheQueue.io_deq_valid && mem_resp.valid && mem_resp.ready) cache_memory_overlap++;
      for (int i = 0; i < `COH_MSHRS; i++) begin
        if (cache_queued[i]) begin
          assert (cache_active[i] && cache_payloads[i] === cache_snapshots[i])
          else $fatal(1, "Queued Cache payload changed for slot %0d", i);
          assert (!(dut.mshr.io_release_valid && dut.mshr.io_release_bits == i))
          else $fatal(1, "Queued Cache slot released: %0d", i);
          assert (!(dut.mshr.io_allocate_valid && dut.mshr.io_allocate_ready && dut.mshr.io_allocatedId == i))
          else $fatal(1, "Queued Cache slot reused: %0d", i);
        end
        if (dut.cacheQueue.io_enq_valid && dut.cacheQueue.io_enq_ready && dut.cacheQueue.io_enq_bits_id == i) begin
          assert (!cache_queued[i])
          else $fatal(1, "Slot already has a queued Cache command: %0d", i);
          cache_snapshots[i] <= cache_payloads[i];
          cache_queued[i] <= 1'b1;
        end
        if (dut.cache.io_request_valid && dut.cache.io_request_ready && dut.cache.io_request_bits_id == i) begin
          assert (cache_queued[i] && dut.cache.io_request_bits_data === cache_snapshots[i])
          else $fatal(1, "Cache command lost its queued payload: %0d", i);
          cache_queued[i] <= 1'b0;
        end
      end
    end
  end
  final
    $display(
        "CACHE_PAYLOAD: enqueues=%0d accepts=%0d stalls=%0d response_overlap=%0d memory_overlap=%0d pending=%h",
        cache_enqueues,
        cache_accepts,
        cache_stalls,
        cache_response_overlap,
        cache_memory_overlap,
        cache_queued
    );
  // Check the transaction view against the live directory throughout its ownership.
  wire [`COH_SETS-1:0][`COH_WAYS-1:0][`COH_AGENTS:0] directory_table = `COH_DIRECTORY_TABLE;
  wire [`COH_MSHRS-1:0][`COH_AGENTS:0] directory_snapshots = `COH_DIRECTORY_SNAPSHOTS;
  wire [`COH_MSHRS-1:0][4:0] directory_states = `COH_DIRECTORY_STATES;
  wire [`COH_MSHRS-1:0][`COH_DIRECTORY_SET_BITS-1:0] directory_sets = `COH_DIRECTORY_SETS;
  wire [`COH_MSHRS-1:0][`COH_DIRECTORY_WAY_BITS-1:0] directory_ways = `COH_DIRECTORY_WAYS;
  longint directory_checks = 0;
  always @(posedge clock) begin
    if (!control.reset) begin
      for (int i = 0; i < `COH_MSHRS; i++) begin
        if (cache_active[i] && directory_states[i] != 0 && directory_states[i] != 1) begin
          assert (directory_snapshots[i] === directory_table[directory_sets[i]][directory_ways[i]])
          else $fatal(1, "Directory snapshot differs from live entry for slot %0d", i);
          directory_checks++;
        end
      end
    end
  end
  final $display("DIRECTORY_SNAPSHOT: live_entry_checks=%0d", directory_checks);
  initial begin
    uvm_config_db#(virtual coherence_control_if)::set(null, "uvm_test_top*", "control", control);
    uvm_config_db#(virtual stream_if #(`COH_REQ_WIDTH))::set(null, "uvm_test_top*", "req",
                                                             source_if);
    uvm_config_db#(virtual stream_if #(`COH_RSP_WIDTH))::set(null, "uvm_test_top*", "rx_rsp",
                                                             rx_rsp);
    uvm_config_db#(virtual stream_if #(`COH_DAT_WIDTH))::set(null, "uvm_test_top*", "rx_dat",
                                                             rx_dat);
    uvm_config_db#(virtual stream_if #(`COH_TXRSP_WIDTH))::set(null, "uvm_test_top*", "tx_rsp",
                                                               tx_rsp);
    uvm_config_db#(virtual stream_if #(`COH_TXDAT_WIDTH))::set(null, "uvm_test_top*", "tx_dat",
                                                               sink_if);
    uvm_config_db#(virtual stream_if #(`COH_SNP_CHANNEL_WIDTH))::set(null, "uvm_test_top*", "snp",
                                                                     tx_snp);
    uvm_config_db#(virtual stream_if #(`COH_MEMREQ_WIDTH))::set(null, "uvm_test_top*", "mem_req",
                                                                mem_req);
    uvm_config_db#(virtual stream_if #(`COH_MEMRESP_WIDTH))::set(null, "uvm_test_top*", "mem_resp",
                                                                 mem_resp);
    run_test("protocol_test");
  end
endmodule
