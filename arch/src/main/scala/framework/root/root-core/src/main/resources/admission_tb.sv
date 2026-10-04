module admission_tb;
  import uvm_pkg::*;
  import admission_pkg::*;
  `include "uvm_macros.svh"
  `include "admission_config.svh"
  `include "admission_fixture.svh"
  `define AF(B, K, F) B[`ADMIT_``K``_``F``_OFFSET +: `ADMIT_``K``_``F``_WIDTH]
  logic clock = 0;
  always #5 clock = ~clock;
  admission_control_if control (clock);
  stream_if #(`ADMIT_PTE_WIDTH) pte_req (
      clock,
      control.reset
  );
  stream_if #(`ADMIT_PTERESP_WIDTH) pte_resp (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_RESERVE_WIDTH)
  ) reserve (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_COMMAND_WIDTH)
  ) command (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_COMPLETE_WIDTH)
  ) complete (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_CANCELLED_WIDTH)
  ) cancelled (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_RESPONSE_WIDTH)
  ) response (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_MAINTENANCE_WIDTH)
  ) maintenance (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_MAINTAINED_WIDTH)
  ) maintained (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_MEMREQ_WIDTH)
  ) mem_req (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_MEMRESP_WIDTH)
  ) mem_resp (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_UNCACHED_WIDTH)
  ) uncached_req (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_URESP_WIDTH)
  ) uncached_resp (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_RESERVE_WIDTH)
  ) info (
      clock,
      control.reset
  );
  stream_if #(
      .WIDTH(`ADMIT_RESERVE_WIDTH)
  ) cancel_request (
      clock,
      control.reset
  );
  logic tracker_complete_valid, tracker_complete_ready, tracker_cancel_ready;
  logic [`ADMIT_COMPLETE_WIDTH-1:0] tracker_complete_tag;
  logic maintenance_valid, grant_valid;
  AdmissionCore dut (
      .clock(clock),
      .reset(control.reset),
      .io_resetVector(64'h80000000),
      .io_timerInterrupt(1'b0),
      .io_softwareInterrupt(1'b0),
      .io_externalInterrupt(1'b0),
      .io_retired(control.retired),
      .io_retiredPc(control.retired_pc),
      .io_trapped(control.trapped),
      .io_trapCause(control.trap_cause),
      .io_trapValue(control.trap_value),
      .io_trapPc(control.trap_pc),
      .io_cancelledData(control.cancelled),
      .io_memoryOutstanding(control.outstanding),
      .io_blockMaintenanceResponse(control.block_maintenance_response),
      .io_maintenanceResponsePending(control.maintenance_response_pending),
      `include "admission_ports.svh"
  );
  // The first opaque command owns one write range; real maintenance uses this Core and Home.
  logic tracker_allow, tracker_maint_ready, tracker_done_ready, tracker_maintained_ready;
  logic [7:0] tracker_grant_tag;
  logic [7:0] tracker_maint_tag;
  logic [43:0] tracker_maint_first, tracker_maint_last;
  logic [1:0] tracker_maint_op;
  logic dma_done_valid = 0;
  logic [7:0] dma_done_tag = 0;
  bit use_tracker_maintenance = 0;
  assign control.cpu_allow = tracker_allow;
  Interlock tracker (
      .clock(clock),
      .reset(control.reset),
      .io_dispatch_valid(reserve.valid),
      .io_dispatch_ready(reserve.ready),
      .io_dispatch_bits_id(reserve.bits),
      .io_cancel_valid(cancel_request.valid && cancelled.ready),
      .io_cancel_ready(tracker_cancel_ready),
      .io_cancel_bits_id(cancel_request.bits),
      .io_accessInfo_valid(info.valid),
      .io_accessInfo_ready(info.ready),
      .io_accessInfo_bits_id(info.bits),
      .io_accessInfo_bits_hasMemory(1'b1),
      .io_accessInfo_bits_base(44'h800040c0),
      .io_accessInfo_bits_bytes(45'd64),
      .io_accessInfo_bits_write(1'b1),
      .io_accessInfo_bits_last(1'b1),
      .io_cpuQuery_valid(`AF(control.query, QUERY, VALID)),
      .io_cpuQuery_paddr(`AF(control.query, QUERY, PADDR)),
      .io_cpuQuery_sizeLog2(`AF(control.query, QUERY, SIZELOG2)),
      .io_cpuQuery_write(`AF(control.query, QUERY, WRITE)),
      .io_cpuQuery_olderDispatchPending(`AF(control.query, QUERY, OLDERDISPATCHPENDING)),
      .io_cpuAllow(tracker_allow),
      .io_maintenance_valid(maintenance_valid),
      .io_maintenance_bits_tag(tracker_maint_tag),
      .io_maintenance_bits_firstLine(tracker_maint_first),
      .io_maintenance_bits_lastLine(tracker_maint_last),
      .io_maintenance_bits_op(tracker_maint_op),
      .io_maintenance_ready(tracker_maint_ready),
      .io_maintained_valid(use_tracker_maintenance && maintained.valid),
      .io_maintained_ready(tracker_maintained_ready),
      .io_maintained_bits_tag(`AF(maintained.bits, MAINTAINED, TAG)),
      .io_maintained_bits_ok(`AF(maintained.bits, MAINTAINED, OK)),
      .io_grant_valid(grant_valid),
      .io_grant_bits_tag(tracker_grant_tag),
      .io_grant_ready(1'b1),
      .io_done_valid(dma_done_valid),
      .io_done_ready(tracker_done_ready),
      .io_done_bits_tag(dma_done_tag),
      .io_done_bits_ok(1'b1),
      .io_complete_valid(tracker_complete_valid),
      .io_complete_ready(tracker_complete_ready),
      .io_complete_bits_tag(tracker_complete_tag)
  );
  assign complete.valid = tracker_complete_valid && !control.hold_complete;
  assign complete.bits = tracker_complete_tag;
  assign tracker_complete_ready = complete.ready && !control.hold_complete;
  assign cancelled.valid = cancel_request.valid && tracker_cancel_ready;
  assign cancelled.bits = cancel_request.bits;
  assign cancel_request.ready = tracker_cancel_ready && cancelled.ready;
  typedef struct {
    int id, due;
    longint unsigned addr;
    bit write;
    bit [511:0] data;
    bit [63:0] mask;
  } memory_entry;
  memory_entry pending_memory[$], active_memory;
  bit
      memory_active = 0,
      uncached_active = 0,
      model_ready = 0,
      old_retired = 0,
      young_retired = 0,
      exit_seen = 0,
      exit_retired = 0;
  bit cmo_done = 0;
  int cmo_requests = 0, cmo_acks = 0, cmo_drain_waits = 0, cmo_late_rsp_waits = 0;
  int
      cycle = 0,
      reservations = 0,
      commands_seen = 0,
      traps = 0,
      completed = 0,
      cancelled_count = 0,
      old_pte_waits = 0;
  logic [`ADMIT_URESP_WIDTH-1:0] uncached_result;
  int uncached_due = 0, uncached_ptes = 0;
  int external_ptes = 0, pte_stalls = 0, grants = 0, dma_done_count = 0, blocked_cmo = 0;
  bit live_tags[`ADMIT_ENTRIES];
  logic [`ADMIT_COMMAND_WIDTH-1:0] snapshots[2];
  chandle memory;
  function automatic void check(bit condition, string message);
    if (!condition) `uvm_fatal("ADMISSION", message)
  endfunction
  always @(posedge clock) begin
    if (model_ready && !control.reset) begin
      cycle++;
      if (pte_req.valid && pte_req.ready) external_ptes++;
      if (pte_resp.valid && !pte_resp.ready) pte_stalls++;
      if (grant_valid) begin
        check(tracker_grant_tag == `AF(snapshots[0], COMMAND, TAG), "wrong DMA grant tag");
        grants++;
      end
      if (dma_done_valid && tracker_done_ready) dma_done_count++;
      if (maintenance.valid && !maintenance.ready) cmo_drain_waits++;
      if (maintenance.valid && maintenance.ready) begin
        check(
            dut.core.cache.io_olderRequestsDrained && !dut.core.cacheOffer && !dut.core.cacheOwner &&
            !dut.core.uncachedOffer && !dut.core.uncachedOwner && !dut.core.pteDenied,
            "CMO accepted before cache/PTE owners drained");
        if (!use_tracker_maintenance) check(dut.core.lsu.io_idle, "old CPU owner bypassed drain");
        else if (dut.core.state != 0 || !dut.core.lsu.io_idle) begin
          check(!control.cpu_allow && `AF(control.query, QUERY, VALID),
                "CMO bypassed a nonblocked busy broker");
          blocked_cmo++;
        end
        check(old_pte_waits > 0, "CMO fixture did not overlap actual page-table read");
        cmo_requests++;
      end
      if (control.maintenance_response_pending && control.block_maintenance_response) begin
        cmo_late_rsp_waits++;
        check(!maintained.valid, "CMO acknowledged before delayed real CHI completion");
      end
      if (maintained.valid) begin
        check(`AF(maintained.bits, MAINTAINED, TAG) == (use_tracker_maintenance ?
              `AF(snapshots[0], COMMAND, TAG) : 8'he1) && `AF(maintained.bits, MAINTAINED, OK),
              "Actual Core CMO failed or changed tag/status");
      end
      if (maintained.valid && maintained.ready) cmo_acks++;
      if (!use_tracker_maintenance)
        check(!maintenance_valid && !grant_valid, "early tracker maintenance/grant");
      if (control.retired) begin
        if (control.retired_pc == `ADMIT_OLD_LOAD_PC) old_retired = 1;
        if (control.retired_pc == `ADMIT_YOUNG_LOAD_PC) begin
          check(commands_seen >= 1, "young load retired before descriptor delivery");
          young_retired = 1;
        end
        if (control.retired_pc == `CORE_EXIT_PC) exit_retired = 1;
      end
      if (control.trapped) begin
        check(traps == 0 && control.trap_cause == 2 && control.trap_pc == `ADMIT_FAULT_PC,
              "unexpected precise trap");
        traps++;
      end
      if(dut.core.lsu.io_cpu_req_valid&&dut.core.lsu.io_cpu_req_ready&&dut.core.lsu.io_pc==`ADMIT_YOUNG_LOAD_PC)
        check(reservations >= 1, "young EX memory accepted ahead of older RoCC reservation");
      if (reserve.valid && reserve.ready) begin
        int tag;
        tag = reserve.bits;
        check(old_retired && tag < `ADMIT_ENTRIES && !live_tags[tag] && reservations < 2,
              "early/duplicate/faulting reservation");
        live_tags[tag] = 1;
        reservations++;
        `uvm_info("RESERVE", $sformatf("tag%0d pc=%h", tag, dut.core.cpu.io_rocc_cmd_bits_pc),
                  UVM_LOW)
      end
      if (command.valid && command.ready) begin
        int tag;
        tag = `AF(command.bits, COMMAND, TAG);
        check(commands_seen < 2 && live_tags[tag], "snapshot has no live reservation");
        snapshots[commands_seen] = command.bits;
        commands_seen++;
        `uvm_info("COMMAND", $sformatf("tag%0d pc=%h satp=%h priv%0d", tag,
                                       `AF(command.bits, COMMAND, INSTRUCTION_PC),
                                       `AF(command.bits, COMMAND, SATP), `AF(command.bits, COMMAND,
                                                                             EFFECTIVEPRIVILEGE)),
                  UVM_LOW)
      end
      if (complete.valid && complete.ready) begin
        check(live_tags[complete.bits], "duplicate completion");
        live_tags[complete.bits] = 0;
        completed++;
      end
      if (cancelled.valid && cancelled.ready) begin
        check(live_tags[cancelled.bits], "duplicate cancellation");
        live_tags[cancelled.bits] = 0;
        cancelled_count++;
      end
      if (mem_resp.valid && mem_resp.ready) begin
        check(memory_active && `AF(mem_resp.bits, MEMRESP, ID) == active_memory.id,
              "DDR response identity changed");
        if (active_memory.write)
          core_ref_write(memory, active_memory.addr, active_memory.data, active_memory.mask);
        memory_active = 0;
      end
      if (mem_req.valid && mem_req.ready) begin
        memory_entry item;
        item.id = `AF(mem_req.bits, MEMREQ, ID);
        item.addr = `AF(mem_req.bits, MEMREQ, ADDR);
        item.write = `AF(mem_req.bits, MEMREQ, WRITE);
        item.data = `AF(mem_req.bits, MEMREQ, DATA);
        item.mask = `AF(mem_req.bits, MEMREQ, MASK);
        if (!item.write) core_ref_read(memory, item.addr, item.data);
        item.due = cycle + (item.addr == 64'h80006000 ? 40 : 9);
        pending_memory.push_back(item);
        if (item.addr == 64'h80006000) old_pte_waits++;
      end
      if (uncached_resp.valid && uncached_resp.ready) uncached_active = 0;
      if (uncached_req.valid && uncached_req.ready) begin
        check(!uncached_active, "uncached owner overlapped");
        uncached_result = '0;
        `AF(uncached_result, URESP, TAG) = `AF(uncached_req.bits, UNCACHED, TAG);
        uncached_due = cycle + 11;
        if (`AF(uncached_req.bits, UNCACHED, ADDR) inside {64'ha0000000, 64'ha0000008}) begin
          check(!`AF(uncached_req.bits, UNCACHED, WRITE) && `AF(uncached_req.bits, UNCACHED, NORMAL)
                && `AF(uncached_req.bits, UNCACHED, ATOMIC) == 0 &&
                `AF(uncached_req.bits, UNCACHED, SIZE) == 3, "external normal PTE request shape");
          uncached_ptes++;
          `AF(uncached_result, URESP, DATA) = 64'hfedcba98765432c1;
          if (`AF(uncached_req.bits, UNCACHED, ADDR) == 64'ha0000008)
            `AF(uncached_result, URESP, ERROR) = 1;
          else `AF(uncached_result, URESP, ERROR) = 0;
        end else begin
          check(`AF(uncached_req.bits, UNCACHED, WRITE) && `AF(uncached_req.bits, UNCACHED, ADDR)
                == 64'h10000000, "unexpected device/PTE side effect");
          check(`AF(uncached_req.bits, UNCACHED, DATA) == 0, "firmware reported admission failure");
          exit_seen = 1;
        end
        uncached_active = 1;
      end
    end
  end
  always @(negedge clock) begin
    if (model_ready && !control.reset) begin
      mem_req.ready = cycle % 5 != 0;
      if (!memory_active) begin
        mem_resp.valid = 0;
        for (int i = pending_memory.size() - 1; i >= 0; i--)
        if (!memory_active && pending_memory[i].due <= cycle) begin
          active_memory = pending_memory[i];
          pending_memory.delete(i);
          memory_active = 1;
          mem_resp.bits = '0;
          `AF(mem_resp.bits, MEMRESP, ID) = active_memory.id;
          `AF(mem_resp.bits, MEMRESP, DATA) = active_memory.write ? '0 : active_memory.data;
          mem_resp.valid = 1;
        end
      end
      uncached_req.ready  = !uncached_active && cycle % 5 != 0;
      uncached_resp.valid = uncached_active && cycle >= uncached_due;
      uncached_resp.bits  = uncached_result;
    end
  end
  task automatic external_pte(input longint unsigned addr, expected, input bit error);
    @(negedge clock);
    pte_req.bits = '0;
    `AF(pte_req.bits, PTE, ADDR) = addr;
    pte_req.valid = 1;
    do @(posedge clock); while (!pte_req.ready);
    @(negedge clock);
    pte_req.valid = 0;
    wait (pte_resp.valid);
    repeat (4) begin
      @(posedge clock);
      check(pte_resp.valid && `AF(pte_resp.bits, PTERESP, ERROR) == error &&
            `AF(pte_resp.bits, PTERESP, DATA) == expected,
            "external PTE response owner/data/error unstable");
    end
    @(negedge clock);
    pte_resp.ready = 1;
    do @(posedge clock); while (!pte_resp.valid);
    @(negedge clock);
    pte_resp.ready = 0;
  endtask
  task automatic check_context(logic [`ADMIT_COMMAND_WIDTH-1:0] packet, bit second);
    check(`AF(packet, COMMAND, INSTRUCTION_OPCODE) == 7'h7b &&
          `AF(packet, COMMAND, INSTRUCTION_FUNCT3) == 3 && `AF(packet, COMMAND, INSTRUCTION_XS1) &&
          `AF(packet, COMMAND, INSTRUCTION_XS2) && !`AF(packet, COMMAND, INSTRUCTION_XD),
          "stock RoCC classification/operands altered");
    check(`AF(packet, COMMAND, INSTRUCTION_PC) == (second ? `ADMIT_CMD1_PC : `ADMIT_CMD0_PC),
          "command PC changed");
    check(`AF(packet, COMMAND, INSTRUCTION_RS1DATA)
          == (second ? 64'h5555666677778888 : 64'h1111222233334444), "rs1 snapshot changed");
    check(`AF(packet, COMMAND, INSTRUCTION_RS2DATA) == (second ? 64'h400040c8 : 64'h400040c0),
          "rs2 snapshot changed");
    check(`AF(packet, COMMAND, SATP) == (second ? 64'h8000300000080009 : 64'h8000000000080006),
          "satp snapshot changed");
    check(`AF(packet, COMMAND, EFFECTIVEPRIVILEGE) == (second ? 3 : 1) && `AF(packet, COMMAND, SUM)
          == !second && `AF(packet, COMMAND, MXR) == !second, "privilege/SUM/MXR snapshot changed");
    check(`AF(packet, COMMAND, PMP_0_ADDR) == (second ? 64'h20800000 : 64'h20400000) &&
          `AF(packet, COMMAND, PMP_0_CFG_A) == 1 && `AF(packet, COMMAND, PMP_0_CFG_R) &&
          `AF(packet, COMMAND, PMP_0_CFG_W) && `AF(packet, COMMAND, PMP_0_CFG_X) == !second && !
          `AF(packet, COMMAND, PMP_0_CFG_L), "PMP snapshot changed");
  endtask
  always @(negedge clock) begin
    if (use_tracker_maintenance) begin
      maintenance.valid = maintenance_valid;
      maintenance.bits = '0;
      `AF(maintenance.bits, MAINTENANCE, TAG) = tracker_maint_tag;
      `AF(maintenance.bits, MAINTENANCE, FIRSTLINE) = tracker_maint_first;
      `AF(maintenance.bits, MAINTENANCE, LASTLINE) = tracker_maint_last;
      `AF(maintenance.bits, MAINTENANCE, OP) = tracker_maint_op;
      maintained.ready = tracker_maintained_ready;
    end
  end
  always_comb tracker_maint_ready = use_tracker_maintenance && maintenance.ready;
  initial begin
    uvm_config_db#(virtual admission_control_if)::set(null, "*", "control", control);
    run_test("protocol_test");
  end
  initial begin
    pte_req.valid = 0;
    pte_req.bits = '0;
    pte_resp.ready = 0;
    maintenance.valid = 0;
    maintenance.bits = '0;
    maintained.ready = 0;
    control.block_maintenance_response = 1;
    control.reset = 1;
    control.accelerator_irq = 0;
    command.ready = 0;
    response.valid = 0;
    response.bits = '0;
    info.valid = 0;
    info.bits = '0;
    cancel_request.valid = 0;
    cancel_request.bits = '0;
    mem_req.ready = 0;
    mem_resp.valid = 0;
    mem_resp.bits = '0;
    uncached_req.ready = 0;
    uncached_resp.valid = 0;
    uncached_resp.bits = '0;
    foreach (live_tags[i]) live_tags[i] = 0;
    wait (control.start);
    memory = core_ref_create(`ADMIT_IMAGE);
    model_ready = 1;
    repeat (4) @(posedge clock);
    @(negedge clock);
    control.reset = 0;
    fork
      begin
        wait (old_pte_waits > 0);
        @(negedge clock);
        maintenance.bits = '0;
        `AF(maintenance.bits, MAINTENANCE, TAG) = 8'he1;
        `AF(maintenance.bits, MAINTENANCE, FIRSTLINE) = 44'h800040c0;
        `AF(maintenance.bits, MAINTENANCE, LASTLINE) = 44'h800040c0;
        `AF(maintenance.bits, MAINTENANCE, OP) = 0;
        maintenance.valid = 1;
        do @(posedge clock); while (!maintenance.ready);
        @(negedge clock);
        maintenance.valid = 0;
        wait (control.maintenance_response_pending);
        repeat (8) begin
          @(posedge clock);
          #1;
          check(!maintained.valid && reservations == 0,
                "Late CMO completion released new admission");
        end
        @(negedge clock);
        control.block_maintenance_response = 0;
        wait (maintained.valid);
        begin
          logic [`ADMIT_MAINTAINED_WIDTH-1:0] held;
          held = maintained.bits;
          repeat (8) begin
            @(posedge clock);
            #1;
            check(maintained.valid && maintained.bits === held && reservations == 0,
                  "Held CMO ACK changed or released new admission");
          end
        end
        @(negedge clock);
        maintained.ready = 1;
        do @(posedge clock); while (!maintained.valid);
        @(negedge clock);
        maintained.ready = 0;
        cmo_done = 1;
      end
    join_none
    wait (command.valid);
    begin
      logic [`ADMIT_COMMAND_WIDTH-1:0] held;
      held = command.bits;
      check_context(held, 0);
      repeat (8) begin
        @(posedge clock);
        #1;
        check(command.valid && command.bits === held && reservations == 1 && !young_retired,
              "snapshot/young request escaped admission stall");
      end
    end
    @(negedge clock);
    command.ready = 1;
    wait (commands_seen == 1);
    @(negedge clock);
    command.ready = 0;
    // A younger authorized physical query waits behind the unknown reservation.
    wait (!control.cpu_allow && `AF(control.query, QUERY, VALID));
    external_pte(64'h80006008, 64'h20001c01, 0);
    external_pte(64'ha0000008, 0, 1);
    external_pte(64'h10000000, 0, 1);
    @(negedge clock);
    pte_req.bits = '0;
    `AF(pte_req.bits, PTE, ADDR) = 44'ha0000000;
    pte_req.valid = 1;
    do @(posedge clock); while (!pte_req.ready);
    @(negedge clock);
    pte_req.valid = 0;
    wait (pte_resp.valid);
    check(!`AF(pte_resp.bits, PTERESP, ERROR) && `AF(pte_resp.bits, PTERESP, DATA)
          == 64'hfedcba98765432c1, "external PTE did not return the uncached normal RAM value");
    use_tracker_maintenance = 1;
    info.bits = `AF(snapshots[0], COMMAND, TAG);
    info.valid = 1;
    do @(posedge clock); while (!info.ready);
    @(negedge clock);
    info.valid = 0;
    wait (maintenance_valid);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(pte_resp.valid && !tracker_maint_ready && !young_retired,
            "held external PTE did not block maintenance or younger access");
    end
    @(negedge clock);
    pte_resp.ready = 1;
    do @(posedge clock); while (!pte_resp.valid);
    @(negedge clock);
    pte_resp.ready = 0;
    wait (grants == 1);
    wait (!control.cpu_allow &&
    `AF(control.query, QUERY, VALID)
    &&
    `AF(control.query, QUERY, PADDR)
    == 44'h800040c8 && dut.core.state != 0);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(!young_retired, "young CPU escaped live write range");
    end
    @(negedge clock);
    control.hold_complete = 0;
    dma_done_tag = `AF(snapshots[0], COMMAND, TAG);
    dma_done_valid = 1;
    do @(posedge clock); while (!tracker_done_ready);
    @(negedge clock);
    dma_done_valid = 0;
    wait (command.valid);
    check_context(command.bits, 1);
    check(completed == 1, "second command proceeded before real first completion");
    @(negedge clock);
    command.ready = 1;
    wait (commands_seen == 2);
    @(negedge clock);
    command.ready = 0;
    cancel_request.bits = `AF(snapshots[1], COMMAND, TAG);
    cancel_request.valid = 1;
    do @(posedge clock); while (!cancel_request.ready);
    @(negedge clock);
    cancel_request.valid  = 0;
    control.hold_complete = 0;
    wait(cmo_done&&exit_seen&&exit_retired&&completed==1&&cancelled_count==1&&control.admission_outstanding==0&&control.outstanding==0);
    repeat (8) @(posedge clock);
    check(
        reservations == 2 && commands_seen == 2 && traps == 1 && old_pte_waits > 0 && young_retired,
        "actual Core admission fixture did not complete required observations");
    check(!memory_active && pending_memory.size() == 0 && !uncached_active,
          "memory responses not drained");
    check(
        cmo_requests == 3 && cmo_acks == 3 && cmo_drain_waits > 0 && cmo_late_rsp_waits >= 8 &&
          external_ptes==4 && uncached_ptes==2 && pte_stalls>=8 && grants==1 && dma_done_count==1 && blocked_cmo>0,
        $sformatf(
        "CMO requirements req%0d ack%0d oldwait%0d latewait%0d pte%0d stalls%0d grant%0d done%0d blocked%0d",
        cmo_requests,
        cmo_acks,
        cmo_drain_waits,
        cmo_late_rsp_waits,
        external_ptes,
        pte_stalls,
        grants,
        dma_done_count,
        blocked_cmo
        ));
    `uvm_info("CORE_MAINTENANCE_PASS", $sformatf(
              "request=%0d ack=%0d olderOwnerWait=%0d lateCHIWait=%0d; actual Core CMO and held ACK checked",
              cmo_requests,
              cmo_acks,
              cmo_drain_waits,
              cmo_late_rsp_waits
              ), UVM_LOW)
    `uvm_info("CHECKED", $sformatf(
              "reserve%0d cmd%0d complete%0d cancel%0d preciseTrap%0d oldPTE%0d",
              reservations,
              commands_seen,
              completed,
              cancelled_count,
              traps,
              old_pte_waits
              ), UVM_LOW)
    core_ref_destroy(memory);
    model_ready = 0;
    control.finished = 1;
  end
endmodule
