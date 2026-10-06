module consistency_tb;
  import uvm_pkg::*;
  import consistency_pkg::*;
  `include "uvm_macros.svh"
  `include "consistency_config.svh"
  `define CF(B, K, F) B[`CONS_``K``_``F``_OFFSET +: `CONS_``K``_``F``_WIDTH]
  logic clock = 0;
  always #5 clock = ~clock;
  consistency_control_if control (clock);
  stream_if #(.WIDTH(`CONS_CPU_WIDTH))
      access_0 (
          clock,
          control.reset
      ),
      access_1 (
          clock,
          control.reset
      );
  stream_if #(.WIDTH(`CONS_RESULT_WIDTH))
      result_0 (
          clock,
          control.reset
      ),
      result_1 (
          clock,
          control.reset
      );
  stream_if #(
      .WIDTH(`CONS_DISPATCH_WIDTH)
  ) dispatch (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`CONS_INFO_WIDTH)
  ) access_info (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`CONS_GRANT_WIDTH)
  ) grant (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`CONS_DONE_WIDTH)
  ) done (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`CONS_COMPLETE_WIDTH)
  ) complete (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`CONS_MEMREQ_WIDTH)
  ) mem_req (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`CONS_MEMRESP_WIDTH)
  ) mem_resp (
      clock,
      control.reset
  );
  Consistency dut (
      .clock(clock),
      .reset(control.reset),
      `include "consistency_ports.svh"
  );
  localparam longint unsigned A='h4000,B='h4040,P='h10000,X='hC000,Y='h8040,E='h20000,R='h30000+3*`CONS_LINE_BYTES;
  localparam longint unsigned EVICT = E + `CONS_L1_LINES * `CONS_LINE_BYTES;
  typedef struct {
    longint unsigned addr, data, expected;
    int unsigned mask;
    bit write;
  } cpu_entry;
  typedef struct {
    int unsigned id;
    longint unsigned addr;
    bit write;
    bit [511:0] data;
    bit [63:0] mask;
    int due;
  } memory_entry;
  cpu_entry cpu_pending[2][$];
  memory_entry memory_pending[$], active_memory;
  bit memory_active = 0, memory_live[`CONS_MSHRS];
  chandle model;
  bit model_ready = 0, hold_write = 0, hold_read = 0, old_fill_seen = 0;
  longint unsigned hold_write_addr, hold_read_addr;
  int cycle = 0, checked[2], submitted[2], grants[256], completions[256];
  int memory_reads = 0, memory_writes = 0, memory_acks = 0, peak = 0, cmo_count[3], peer_snoops = 0;
  function automatic void check(bit condition, string message);
    if (!condition) `uvm_fatal("CONSISTENCY", message)
  endfunction
  function automatic void observe_cpu(int core, bit valid, ready,
                                      logic [`CONS_CPU_WIDTH-1:0] request, bit response_valid,
                                      response_ready, logic [`CONS_RESULT_WIDTH-1:0] response);
    if (valid && ready) begin
      cpu_entry item;
      check(!$isunknown(request), "unknown CPU request");
      check(`CF(request, CPU, ATOMIC) == 0 && !`CF(request, CPU, ATOMICWORD),
            "physical consistency profile uses ordinary 64-bit CPU accesses");
      item.addr = `CF(request, CPU, ADDR);
      item.data = `CF(request, CPU, DATA);
      item.mask = `CF(request, CPU, MASK);
      item.write = `CF(request, CPU, WRITE);
      item.expected = cons_ref_expected64(model, item.addr);
      cpu_pending[core].push_back(item);
      submitted[core]++;
      `uvm_info("CPU_ACCEPT", $sformatf("core%0d addr=%h write%0d", core, item.addr, item.write),
                UVM_LOW)
    end
    if (response_valid && response_ready) begin
      cpu_entry item;
      check(cpu_pending[core].size() != 0, "CPU response without accepted access");
      item = cpu_pending[core].pop_front();
      check(!$isunknown(response) && !`CF(response, RESULT, ERROR), "CPU memory access failed");
      check(`CF(response, RESULT, DATA) === item.expected, $sformatf(
            "core%0d addr=%h result=%h expected=%h",
            core,
            item.addr,
            `CF(response, RESULT, DATA),
            item.expected
            ));
      if (item.write) cons_ref_cpu_commit(model, item.addr, item.data, item.mask);
      checked[core]++;
      `uvm_info("CPU_RESULT", $sformatf("core%0d addr=%h data=%h", core, item.addr,
                                        `CF(response, RESULT, DATA)), UVM_LOW)
    end
  endfunction
  always @(posedge clock) begin
    if (model_ready && !control.reset) begin
      cycle++;
      if (control.outstanding > peak) peak = control.outstanding;
      observe_cpu(0, access_0.valid, access_0.ready, access_0.bits, result_0.valid, result_0.ready,
                  result_0.bits);
      observe_cpu(1, access_1.valid, access_1.ready, access_1.bits, result_1.valid, result_1.ready,
                  result_1.bits);
      if (dispatch.valid && dispatch.ready) begin
        int id;
        id = `CF(dispatch.bits, DISPATCH, ID);
        grants[id] = 0;
        completions[id] = 0;
      end
      if (grant.valid && grant.ready) begin
        grants[`CF(grant.bits, GRANT, TAG)]++;
        `uvm_info("NPU_GRANT", $sformatf("tag%0d", `CF(grant.bits, GRANT, TAG)), UVM_LOW)
      end
      if (complete.valid && complete.ready) begin
        completions[`CF(complete.bits, COMPLETE, TAG)]++;
        `uvm_info("NPU_COMPLETE", $sformatf("tag%0d", `CF(complete.bits, COMPLETE, TAG)), UVM_LOW)
      end
      if (control.req_valid)
        `uvm_info("HOME_REQUEST", $sformatf(
                  "node%0d txn%0d op%h addr=%h",
                  `CF(control.req_bits, REQ, SRCID),
                  `CF(control.req_bits, REQ, TXNID),
                  `CF(control.req_bits, REQ, OPCODE),
                  `CF(control.req_bits, REQ, ADDR)
                  ), UVM_LOW)
      if (control.snp_valid)
        `uvm_info("HOME_SNOOP", $sformatf(
                  "node%0d op%h addr=%h",
                  control.snp_bits[`CONS_SNP_TARGET_OFFSET+:`CONS_SNP_TARGET_WIDTH],
                  `CF(control.snp_bits, SNP, OPCODE),
                  64'(
                  `CF(control.snp_bits, SNP, ADDR)
                  ) << 3
                  ), UVM_LOW)
      if (control.req_valid &&
          `CF(control.req_bits, REQ, SRCID)
          == 1 &&
          `CF(control.req_bits, REQ, TXNID)
          == 2) begin
        int opcode;
        opcode = `CF(control.req_bits, REQ, OPCODE);
        check(opcode inside {'h08, 'h09, 'h0a}, "maintenance requester emitted a non-CMO opcode");
        check(`CF(control.req_bits, REQ, TGTID) == `CONS_HOME, "CMO routed to wrong Home");
        cmo_count[opcode-'h08]++;
        `uvm_info("CMO_REQUEST", $sformatf("op%h addr=%h", opcode, `CF(control.req_bits, REQ,
                                                                       ADDR)), UVM_LOW)
      end
      if(control.snp_valid&&control.snp_bits[`CONS_SNP_TARGET_OFFSET+:`CONS_SNP_TARGET_WIDTH]==2)
        peer_snoops++;
      if (mem_resp.valid && mem_resp.ready) begin
        check(memory_active && `CF(mem_resp.bits, MEMRESP, ID) == active_memory.id,
              "memory response identity mismatch");
        if (active_memory.write)
          cons_ref_ddr_commit(model, active_memory.addr, active_memory.data, active_memory.mask);
        memory_live[active_memory.id] = 0;
        memory_active = 0;
        memory_acks++;
        `uvm_info("DDR_ACK", $sformatf("id%0d addr=%h write%0d", active_memory.id,
                                       active_memory.addr, active_memory.write), UVM_LOW)
      end
      if (mem_req.valid && mem_req.ready) begin
        memory_entry item;
        check(!$isunknown(mem_req.bits), "unknown real Home memory request");
        item.id = `CF(mem_req.bits, MEMREQ, ID);
        item.addr = `CF(mem_req.bits, MEMREQ, ADDR);
        item.write = `CF(mem_req.bits, MEMREQ, WRITE);
        item.data = `CF(mem_req.bits, MEMREQ, DATA);
        item.mask = `CF(mem_req.bits, MEMREQ, MASK);
        check(item.id < `CONS_MSHRS && !memory_live[item.id], "Home reused a live DDR ID");
        check((item.addr & 63) == 0, "Home DDR request not a 64-byte line");
        memory_live[item.id] = 1;
        if (item.write) begin
          check(item.mask === '1 && cons_ref_write_matches(model, item.addr, item.data, item.mask
                ) == 1, "Home writeback lost CPU-visible dirty bytes");
          memory_writes++;
        end else begin
          cons_ref_ddr_read(model, item.addr, item.data);
          memory_reads++;
          if (item.addr == X) old_fill_seen = 1;
        end
        item.due = cycle + 7 + (3 - item.id) * 3;
        memory_pending.push_back(item);
        `uvm_info("DDR_REQUEST", $sformatf("id%0d addr=%h write%0d", item.id, item.addr,
                                           item.write), UVM_LOW)
      end
    end
  end
  // DDR BFM: reads and write ACKs can reorder across distinct real Home MSHR IDs.
  always @(negedge clock) begin
    if (model_ready && !control.reset) begin
      mem_req.ready = (cycle % 7) != 0;
      if (!memory_active) begin
        mem_resp.valid = 0;
        for (int i = memory_pending.size() - 1; i >= 0; i--) begin
          if(!memory_active&&memory_pending[i].due<=cycle&&
              !(hold_write&&memory_pending[i].write&&memory_pending[i].addr==hold_write_addr)&&
              !(hold_read&&!memory_pending[i].write&&memory_pending[i].addr==hold_read_addr)) begin
            active_memory = memory_pending[i];
            memory_pending.delete(i);
            memory_active = 1;
            mem_resp.bits = '0;
            `CF(mem_resp.bits, MEMRESP, ID) = active_memory.id;
            `CF(mem_resp.bits, MEMRESP, DATA) = active_memory.write ? '0 : active_memory.data;
            mem_resp.valid = 1;
          end
        end
      end
    end
  end
  task automatic cpu_request(int core, longint unsigned address, bit write = 0,
                             longint unsigned data = 0, int unsigned mask = 255);
    logic [`CONS_CPU_WIDTH-1:0] packet;
    packet = '0;
    `CF(packet, CPU, ADDR) = address;
    `CF(packet, CPU, WRITE) = write;
    `CF(packet, CPU, DATA) = data;
    `CF(packet, CPU, MASK) = write ? mask : 0;
    @(negedge clock);
    if (core == 0) begin
      access_0.bits  = packet;
      access_0.valid = 1;
      do @(posedge clock); while (!access_0.ready);
      @(negedge clock);
      access_0.valid = 0;
    end else begin
      access_1.bits  = packet;
      access_1.valid = 1;
      do @(posedge clock); while (!access_1.ready);
      @(negedge clock);
      access_1.valid = 0;
    end
  endtask
  task automatic cpu_access(int core, longint unsigned address, bit write = 0,
                            longint unsigned data = 0, int unsigned mask = 255);
    int goal;
    goal = checked[core] + 1;
    cpu_request(core, address, write, data, mask);
    wait (checked[core] == goal);
    @(negedge clock);
  endtask
  task automatic reserve_access(int tag, longint unsigned base, bytes, bit write);
    @(negedge clock);
    dispatch.bits = '0;
    `CF(dispatch.bits, DISPATCH, ID) = tag;
    dispatch.valid = 1;
    do @(posedge clock); while (!dispatch.ready);
    @(negedge clock);
    dispatch.valid = 0;
    access_info.bits = '0;
    `CF(access_info.bits, INFO, ID) = tag;
    `CF(access_info.bits, INFO, HASMEMORY) = 1;
    `CF(access_info.bits, INFO, BASE) = base;
    `CF(access_info.bits, INFO, BYTES) = bytes;
    `CF(access_info.bits, INFO, WRITE) = write;
    `CF(access_info.bits, INFO, LAST) = 1;
    access_info.valid = 1;
    do @(posedge clock); while (!access_info.ready);
    @(negedge clock);
    access_info.valid = 0;
  endtask
  task automatic dma_done(int tag);
    @(negedge clock);
    done.bits = '0;
    `CF(done.bits, DONE, TAG) = tag;
    `CF(done.bits, DONE, OK) = 1;
    done.valid = 1;
    do @(posedge clock); while (!done.ready);
    @(negedge clock);
    done.valid = 0;
    done.bits  = '0;
  endtask
  task automatic await_complete(int tag);
    wait (completions[tag] == 1);
    @(negedge clock);
  endtask
  task automatic block_cpu_read(longint unsigned address, int clocks = 32);
    @(negedge clock);
    access_0.bits = '0;
    `CF(access_0.bits, CPU, ADDR) = address;
    access_0.valid = 1;
    repeat (clocks) begin
      @(posedge clock);
      #1;
      check(!control.cpu_allow && !access_0.ready,
            "CPU target read passed before DMA ACK/post-invalidate");
    end
  endtask
  task automatic release_cpu_read();
    int goal;
    goal = checked[0] + 1;
    do @(posedge clock); while (!access_0.ready);
    @(negedge clock);
    access_0.valid = 0;
    wait (checked[0] == goal);
    @(negedge clock);
  endtask
  initial begin
    uvm_config_db#(virtual consistency_control_if)::set(null, "*", "control", control);
    run_test("protocol_test");
  end
  initial begin
    control.reset = 1;
    control.older_dispatch_pending = 0;
    control.older_requests_drained = 1;
    control.block_requester_rsp = 0;
    control.block_requester_data = 0;
    access_0.valid = 0;
    access_0.bits = '0;
    access_1.valid = 0;
    access_1.bits = '0;
    result_0.ready = 1;
    result_1.ready = 1;
    dispatch.valid = 0;
    dispatch.bits = '0;
    access_info.valid = 0;
    access_info.bits = '0;
    grant.ready = 1;
    done.valid = 0;
    done.bits = '0;
    complete.ready = 1;
    mem_req.ready = 0;
    mem_resp.valid = 0;
    mem_resp.bits = '0;
    foreach (memory_live[i]) memory_live[i] = 0;
    foreach (checked[i]) begin
      checked[i]   = 0;
      submitted[i] = 0;
    end
    foreach (grants[i]) begin
      grants[i] = 0;
      completions[i] = 0;
    end
    foreach (cmo_count[i]) cmo_count[i] = 0;
    wait (control.start);
    check(
        `CONS_AGENTS==2&&`CONS_SLOTS==4&&`CONS_PA_BITS==44&&`CONS_LINE_BYTES==64&&`CONS_L1_LINES==8&&`CONS_L2_SETS==16,
        "unexpected frozen physical profile");
    model = cons_ref_create();
    model_ready = 1;
    repeat (4) @(posedge clock);
    @(negedge clock);
    control.reset = 0;
    // CPU publication hint stalls only until the older unregistered dispatch is recorded.
    @(negedge clock);
    control.older_dispatch_pending = 1;
    access_0.bits = '0;
    `CF(access_0.bits, CPU, ADDR) = B;
    access_0.valid = 1;
    repeat (8) begin
      @(posedge clock);
      #1;
      check(!control.cpu_allow && !access_0.ready, "older dispatch hint did not block CPU");
    end
    @(negedge clock);
    control.older_dispatch_pending = 0;
    release_cpu_read();
    // Real dirty L1 data must reach DDR before a mock NPU read is granted.
    cpu_access(0, A, 1, 64'h1122334455667788);
    check(cons_ref_ddr_visible(model, A) == 0, "CPU dirty store incorrectly updated DDR golden");
    hold_write = 1;
    hold_write_addr = A;
    reserve_access(10, A, 64, 0);
    wait (memory_writes > 0 || grants[10] > 0);
    check(memory_writes > 0, "dirty pre-Clean granted without a real DDR writeback");
    repeat (32) begin
      @(posedge clock);
      #1;
      check(grants[10] == 0 && completions[10] == 0, "NPU read granted before real DDR write ACK");
    end
    cpu_access(1, B);  // Independent requester/range progresses while dirty write ACK is held.
    @(negedge clock);
    hold_write = 0;
    wait (grants[10] == 1);
    check(cons_ref_ddr_visible(model, A) == 1, "pre-Clean did not publish dirty CPU bytes to DDR");
    cpu_access(0, A);  // R/R overlap completes while NPU read has not sent done.
    check(completions[10] == 0, "NPU read unexpectedly completed without done");
    dma_done(10);
    await_complete(10);
    // Preserve dirty neighbour bytes around a partial NPU write and invalidate all real holders.
    cpu_access(0, P, 1, 64'h0102030405060708);
    cpu_access(0, P + 16, 1, 64'h9192939495969798);
    cpu_access(1, P);
    cpu_access(1, P + 16);
    control.older_requests_drained = 0;
    reserve_access(20, P, 64, 1);
    begin
      int prior_cmo_count;
      prior_cmo_count = cmo_count[1];
      repeat (12) begin
        @(posedge clock);
        #1;
        check(grants[20] == 0 && cmo_count[1] == prior_cmo_count,
              "maintenance admitted before external old requests drained");
      end
    end
    @(negedge clock);
    control.older_requests_drained = 1;
    wait (grants[20] == 1);
    check(cons_ref_ddr_visible(model, P) == 1, "pre-CleanInvalidate lost dirty neighbour bytes");
    cons_ref_dma_write(model, P + 8, 64'hfeedface89abcdef, 15);
    block_cpu_read(P + 8, 32);
    check(completions[20] == 0, "write completed before DMA done ACK");
    @(negedge clock);
    complete.ready = 0;
    dma_done(20);
    wait (complete.valid);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(!control.cpu_allow && !access_0.ready && complete.valid &&
            `CF(complete.bits, COMPLETE, TAG) == 20,
            "CPU read bypassed stalled final range completion");
    end
    @(negedge clock);
    complete.ready = 1;
    await_complete(20);
    release_cpu_read();
    for (int word = 0; word < 8; word++) begin
      cpu_access(0, P + word * 8);
      cpu_access(1, P + word * 8);
    end
    // An old dirty CopyBack can leave L1 yet remain in the explicit transport FIFO.
    cpu_access(0, E, 1, 64'h12345678badc0ffe);
    @(negedge clock);
    control.block_requester_data = 1;
    cpu_access(0, EVICT);  // Same L1 slot, different real Home set.
    check(cons_ref_ddr_visible(model, E) == 0,
          "old dirty CopyBack reached DDR despite transport stall");
    reserve_access(40, E, 64, 0);
    repeat (32) begin
      @(posedge clock);
      #1;
      check(grants[40] == 0 && completions[40] == 0, "CMO granted ahead of old delayed CopyBack");
    end
    @(negedge clock);
    control.block_requester_data = 0;
    wait (grants[40] == 1);
    check(cons_ref_ddr_visible(model, E) == 1, "old CopyBack did not publish DDR before grant");
    dma_done(40);
    await_complete(40);
    // A CompAck accepted by L1 can still be queued at the real Home RX credit boundary.
    @(negedge clock);
    control.block_requester_rsp = 1;
    cpu_access(0, R);
    reserve_access(50, R, 64, 0);
    repeat (24) begin
      @(posedge clock);
      #1;
      check(grants[50] == 0 && completions[50] == 0, "CMO overtook old undelivered CompAck");
    end
    @(negedge clock);
    control.block_requester_rsp = 0;
    wait (grants[50] == 1);
    check(cons_ref_ddr_visible(model, R) == 1, "CompAck drain changed read-only DDR data");
    dma_done(50);
    await_complete(50);
    // Delay an actual older CPU miss/fill and its result, then admit maintenance only after both drain.
    hold_read = 1;
    hold_read_addr = X;
    result_0.ready = 0;
    control.older_requests_drained = 0;
    cpu_request(0, X);
    wait (old_fill_seen);
    reserve_access(30, X, 64, 1);
    cpu_access(1, Y);
    repeat (16) begin
      @(posedge clock);
      #1;
      check(grants[30] == 0, "maintenance overtook an older delayed fill");
    end
    @(negedge clock);
    hold_read = 0;
    wait (result_0.valid);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(grants[30] == 0, "maintenance overtook unaccepted older CPU result");
    end
    @(negedge clock);
    result_0.ready = 1;
    wait (checked[0] == submitted[0]);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(grants[30] == 0, "maintenance ignored olderRequestsDrained=0 after fill");
    end
    @(negedge clock);
    control.older_requests_drained = 1;
    wait (grants[30] == 1);
    check(cons_ref_ddr_visible(model, X) == 1, "old-fill pre-maintenance altered DDR bytes");
    cons_ref_dma_write(model, X, 64'hd0d1d2d3d4d5d6d7, 255);
    block_cpu_read(X, 32);
    dma_done(30);
    await_complete(30);
    release_cpu_read();
    cpu_access(1, X);
    wait (control.outstanding == 0 && !memory_active && memory_pending.size() == 0);
    repeat (8) @(posedge clock);
    check(
        checked[0]==submitted[0]&&checked[1]==submitted[1]&&cpu_pending[0].size()==0&&cpu_pending[1].size()==0,
        "unmatched CPU accesses remain");
    check(
        memory_writes>0&&cmo_count[0]>0&&cmo_count[1]>=2&&cmo_count[2]>=2&&peer_snoops>0,
        "required real CMO/writeback/holder snoops were absent");
    check(peak >= 2, "independent old-fill/peer traffic did not exercise multiple Home MSHRs");
    `uvm_info("CHECKED", $sformatf(
              "CPU%0d/%0d DDRread%0d/write%0d/ACK%0d CMO%0d/%0d/%0d peerSnp%0d peak%0d",
              checked[0],
              checked[1],
              memory_reads,
              memory_writes,
              memory_acks,
              cmo_count[0],
              cmo_count[1],
              cmo_count[2],
              peer_snoops,
              peak
              ), UVM_LOW)
    cons_ref_destroy(model);
    model_ready = 0;
    control.finished = 1;
  end
endmodule
