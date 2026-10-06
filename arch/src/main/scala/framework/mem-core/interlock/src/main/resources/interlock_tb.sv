module interlock_tb;
  import uvm_pkg::*;
  import interlock_pkg::*;
  `include "uvm_macros.svh"
  `include "interlock_config.svh"
  `define ILF(B, K, F) B[`INTERLOCK_``K``_``F``_OFFSET +: `INTERLOCK_``K``_``F``_WIDTH]
  logic clock = 0, reset = 1, cpu_allow;
  always #5 clock = ~clock;
  interlock_control_if ctl (clock);
  stream_if #(
      .WIDTH(`INTERLOCK_DISPATCH_WIDTH)
  ) dispatch (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_CANCEL_WIDTH)
  ) cancel (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_ACCESS_INFO_WIDTH)
  ) access_info (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_MAINTENANCE_WIDTH)
  ) maintenance (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_MAINTAINED_WIDTH)
  ) maintained (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_GRANT_WIDTH)
  ) grant (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_DONE_WIDTH)
  ) done (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_COMPLETE_WIDTH)
  ) complete (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_CPU_QUERY_WIDTH)
  ) cpu_query (
      clock,
      reset
  );
  Interlock dut (
      .clock(clock),
      .reset(reset),
      `include "interlock_ports.svh"
  );
  chandle model;
  int grants[256], completions[256], maintenance_ops[256][3];
  int checked_queries = 0, reserved = 0, acknowledged = 0, released = 0;
  bit model_ready = 0;
  int cache_pending_refills = 0;
  function automatic void check(bit value, string message);
    if (!value) `uvm_fatal("INTERLOCK_CONTRACT", message)
  endfunction
  always @(posedge clock) begin
    if (model_ready) begin
      if (reset) begin
        interlock_ref_reset(model);
        foreach (grants[i]) begin
          grants[i] = 0;
          completions[i] = 0;
          for (int j = 0; j < 3; j++) maintenance_ops[i][j] = 0;
        end
      end else begin
        if (`ILF(cpu_query.bits, CPU_QUERY, VALID)) begin
          check(cpu_allow === interlock_ref_cpu_allow(
                model,
                `ILF(cpu_query.bits, CPU_QUERY, PADDR),
                1 <<
                `ILF(cpu_query.bits, CPU_QUERY, SIZELOG2),
                `ILF(cpu_query.bits, CPU_QUERY, WRITE),
                `ILF(cpu_query.bits, CPU_QUERY, OLDERDISPATCHPENDING),
                dispatch.valid && dispatch.ready
                ), "CPU permit disagrees with transaction oracle");
          checked_queries++;
        end
        if (dispatch.valid && dispatch.ready) begin
          int id;
          id = `ILF(dispatch.bits, DISPATCH, ID);
          interlock_ref_reserve(model, id);
          reserved++;
          grants[id] = 0;
          completions[id] = 0;
          for (int j = 0; j < 3; j++) maintenance_ops[id][j] = 0;
        end
        if (cancel.valid && cancel.ready)
          interlock_ref_cancel(model, `ILF(cancel.bits, CANCEL, ID));
        if (access_info.valid && access_info.ready)
          interlock_ref_info(model, `ILF(access_info.bits, ACCESS_INFO, ID),
                             `ILF(access_info.bits, ACCESS_INFO, HASMEMORY),
                             `ILF(access_info.bits, ACCESS_INFO, BASE),
                             `ILF(access_info.bits, ACCESS_INFO, BYTES),
                             `ILF(access_info.bits, ACCESS_INFO, WRITE),
                             `ILF(access_info.bits, ACCESS_INFO, LAST));
        if (maintenance.valid && maintenance.ready) begin
          interlock_ref_maint_offer(model, `ILF(maintenance.bits, MAINTENANCE, TAG),
                                    `ILF(maintenance.bits, MAINTENANCE, OP),
                                    `ILF(maintenance.bits, MAINTENANCE, FIRSTLINE),
                                    `ILF(maintenance.bits, MAINTENANCE, LASTLINE));
          maintenance_ops[
          `ILF(maintenance.bits, MAINTENANCE, TAG)
          ][
          `ILF(maintenance.bits, MAINTENANCE, OP)
          ]++;
        end
        if (maintained.valid && maintained.ready) begin
          check(`ILF(maintained.bits, MAINTAINED, OK), "normal cache BFM failed ACK");
          interlock_ref_maint_ack(model, `ILF(maintained.bits, MAINTAINED, TAG));
        end
        if (grant.valid && grant.ready) begin
          interlock_ref_grant(model, `ILF(grant.bits, GRANT, TAG));
          grants[`ILF(grant.bits, GRANT, TAG)]++;
        end
        if (done.valid && done.ready) begin
          check(`ILF(done.bits, DONE, OK), "normal DMA BFM failed ACK");
          interlock_ref_dma_ack(model, `ILF(done.bits, DONE, TAG));
          acknowledged++;
        end
        if (complete.valid && complete.ready) begin
          interlock_ref_complete(model, `ILF(complete.bits, COMPLETE, TAG));
          completions[`ILF(complete.bits, COMPLETE, TAG)]++;
          released++;
        end
      end
    end
  end
  task automatic query(longint unsigned addr, bit write, bit permit, int cycles = 3, bit older = 0,
                       int size_log2 = 3);
    @(negedge clock);
    cpu_query.bits = '0;
    `ILF(cpu_query.bits, CPU_QUERY, VALID) = 1;
    `ILF(cpu_query.bits, CPU_QUERY, PADDR) = addr;
    `ILF(cpu_query.bits, CPU_QUERY, SIZELOG2) = size_log2;
    `ILF(cpu_query.bits, CPU_QUERY, WRITE) = write;
    `ILF(cpu_query.bits, CPU_QUERY, OLDERDISPATCHPENDING) = older;
    #1;
    check(cpu_allow === permit, "directed CPU query incorrect or exceeded combinational timing");
    repeat (cycles) begin
      @(posedge clock);
      #1;
      check(cpu_allow === permit, "CPU permit changed during directed query");
    end
    @(negedge clock);
    cpu_query.bits = '0;
  endtask
  task automatic reserve(int id);
    @(negedge clock);
    dispatch.bits = '0;
    `ILF(dispatch.bits, DISPATCH, ID) = id;
    dispatch.valid = 1;
    do @(posedge clock); while (!dispatch.ready);
    @(negedge clock);
    dispatch.valid = 0;
  endtask
  task automatic declare_access(int id, bit memory, longint unsigned base, bytes, bit write,
                                bit last = 1);
    @(negedge clock);
    access_info.bits = '0;
    `ILF(access_info.bits, ACCESS_INFO, ID) = id;
    `ILF(access_info.bits, ACCESS_INFO, HASMEMORY) = memory;
    `ILF(access_info.bits, ACCESS_INFO, BASE) = base;
    `ILF(access_info.bits, ACCESS_INFO, BYTES) = bytes;
    `ILF(access_info.bits, ACCESS_INFO, WRITE) = write;
    `ILF(access_info.bits, ACCESS_INFO, LAST) = last;
    access_info.valid = 1;
    do @(posedge clock); while (!access_info.ready);
    @(negedge clock);
    access_info.valid = 0;
  endtask
  // External cache BFM: maintenance ACK is delayed until old request/refill work drains.
  task automatic maintain(int expected_tag = -1, int expected_op = -1, int old_refill_delay = 7);
    int tag, op;
    logic [`INTERLOCK_MAINTENANCE_WIDTH-1:0] packet;
    wait (maintenance.valid);
    packet = maintenance.bits;
    tag = `ILF(packet, MAINTENANCE, TAG);
    op = `ILF(packet, MAINTENANCE, OP);
    if (expected_tag >= 0) check(tag == expected_tag, "unexpected maintenance owner");
    if (expected_op >= 0) check(op == expected_op, "unexpected maintenance operation");
    @(negedge clock);
    cpu_query.bits = '0;
    `ILF(cpu_query.bits, CPU_QUERY, VALID) = 1;
    `ILF(cpu_query.bits, CPU_QUERY, PADDR) = `ILF(packet, MAINTENANCE, FIRSTLINE);
    `ILF(cpu_query.bits, CPU_QUERY, SIZELOG2) = 3;
    `ILF(cpu_query.bits, CPU_QUERY, WRITE) = (op == 0);
    repeat (5) begin
      @(posedge clock);
      #1;
      check(maintenance.valid && maintenance.bits === packet,
            "maintenance offer changed under backpressure");
    end
    @(negedge clock);
    maintenance.ready = 1;
    @(posedge clock);
    check(maintenance.valid, "maintenance request disappeared");
    @(negedge clock);
    maintenance.ready = 0;
    cache_pending_refills = 1;
    // Independent external refill work completes later; ACK waits for that event.
    fork
      begin
        repeat (old_refill_delay) @(posedge clock);
        #1;
        cache_pending_refills = 0;
      end
    join_none
    wait (cache_pending_refills == 0);
    @(negedge clock);
    check(!maintained.valid, "cache ACK preceded old refill drain");
    maintained.bits = '0;
    `ILF(maintained.bits, MAINTAINED, TAG) = tag;
    `ILF(maintained.bits, MAINTAINED, OK) = 1;
    maintained.valid = 1;
    do @(posedge clock); while (!maintained.ready);
    @(negedge clock);
    maintained.valid = 0;
    maintained.bits  = '0;
    cpu_query.bits   = '0;
  endtask
  task automatic finish_dma(int id);
    @(negedge clock);
    done.bits = '0;
    `ILF(done.bits, DONE, TAG) = id;
    `ILF(done.bits, DONE, OK) = 1;
    done.valid = 1;
    do @(posedge clock); while (!done.ready);
    @(negedge clock);
    done.valid = 0;
    done.bits  = '0;
  endtask
  task automatic await_grant(int id);
    wait (grants[id] == 1);
    @(negedge clock);
  endtask
  task automatic await_complete(int id);
    wait (completions[id] == 1);
    @(negedge clock);
  endtask
  task automatic joint_reset();
    @(negedge clock);
    reset = 1;
    dispatch.valid = 0;
    access_info.valid = 0;
    maintained.valid = 0;
    done.valid = 0;
    maintenance.ready = 0;
    grant.ready = 0;
    complete.ready = 0;
    cpu_query.bits = '0;
    repeat (4) @(posedge clock);
    @(negedge clock);
    reset = 0;
  endtask
  initial begin
    uvm_config_db#(virtual interlock_control_if)::set(null, "*", "vif", ctl);
    run_test("protocol_test");
  end
  initial begin
    dispatch.valid = 0;
    dispatch.bits = '0;
    cancel.valid = 0;
    cancel.bits = '0;
    access_info.valid = 0;
    access_info.bits = '0;
    maintenance.ready = 0;
    maintained.valid = 0;
    maintained.bits = '0;
    grant.ready = 0;
    done.valid = 0;
    done.bits = '0;
    complete.ready = 0;
    cpu_query.valid = 0;
    cpu_query.ready = 0;
    cpu_query.bits = '0;
    wait (ctl.start);
    check(
        `INTERLOCK_ENTRIES==4&&`INTERLOCK_ADDRESS_BITS==44&&`INTERLOCK_ID_BITS==8&&`INTERLOCK_LINE_BYTES==64,
        "unexpected declared verification profile");
    model = interlock_ref_create(`INTERLOCK_ENTRIES, `INTERLOCK_ADDRESS_BITS, `INTERLOCK_LINE_BYTES,
                                 `INTERLOCK_MAX_RANGES);
    model_ready = 1;
    joint_reset();
    query('h4000, 0, 0, 4, 1);
    query('h4000, 0, 1);
    query(0, 1, 1);
    // Same-cycle dispatch must deny even a read/read CPU query until access info is known.
    @(negedge clock);
    dispatch.bits = '0;
    `ILF(dispatch.bits, DISPATCH, ID) = 255;
    dispatch.valid = 1;
    cpu_query.bits = '0;
    `ILF(cpu_query.bits, CPU_QUERY, VALID) = 1;
    `ILF(cpu_query.bits, CPU_QUERY, PADDR) = 'h4000;
    `ILF(cpu_query.bits, CPU_QUERY, SIZELOG2) = 3;
    #1;
    check(!cpu_allow, "same-cycle dispatch was invisible to CPU");
    @(posedge clock);
    check(dispatch.ready, "first reservation blocked");
    @(negedge clock);
    dispatch.valid = 0;
    cpu_query.bits = '0;
    query('h9000, 0, 0, 6);
    declare_access(255, 0, 0, 0, 0);
    wait (complete.valid);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(complete.valid && `ILF(complete.bits, COMPLETE, TAG) == 255,
            "non-memory completion unstable");
    end
    @(negedge clock);
    complete.ready = 1;
    await_complete(255);
    check(grants[255] == 0, "non-memory registration was sent to DMA");
    query('h9000, 1, 1);
    // Fill all register entries, show R/R parallel grants and reverse-order DMA retirement.
    @(negedge clock);
    complete.ready = 0;
    for (int id = 1; id <= 4; id++) begin
      reserve(id);
      declare_access(id, 1, 'h4000, 128, 0);
    end
    query('h4000, 0, 1);
    query('h4000, 1, 0);
    query('h4080, 1, 1);
    @(negedge clock);
    dispatch.valid = 1;
    `ILF(dispatch.bits, DISPATCH, ID) = 5;
    repeat (8) begin
      @(posedge clock);
      #1;
      check(!dispatch.ready, "full table accepted a fifth reservation");
    end
    // Keep the fifth offer valid until a real slot is released and it handshakes.
    // Grant offers must remain stable while the external DMA consumer stalls.
    maintain();
    wait (grant.valid);
    repeat (8) begin
      @(posedge clock);
      #1;
      check(grant.valid, "grant vanished under backpressure");
    end
    @(negedge clock);
    grant.ready = 1;
    for (int i = 0; i < 3; i++) maintain();
    for (int id = 1; id <= 4; id++) await_grant(id);
    @(negedge clock);
    complete.ready = 1;
    finish_dma(4);
    await_complete(4);
    do @(posedge clock); while (!dispatch.ready);
    @(negedge clock);
    dispatch.valid = 0;
    declare_access(5, 0, 0, 0, 0);
    await_complete(5);
    for (int id = 3; id >= 1; id--) begin
      finish_dma(id);
      await_complete(id);
    end
    query('h4000, 1, 1);
    check(interlock_ref_live(model) == 0, "R/R retirement leaked reservations");
    // RAW: CPU read and dependent NPU read must wait for write ACK AND post-invalidate.
    reserve(20);
    declare_access(20, 1, 'h6003, 61, 1);
    reserve(21);
    declare_access(21, 1, 'h6000, 64, 0);
    query('h6000, 0, 0);
    query('h6040, 1, 1);
    maintain(20, 1, 15);
    await_grant(20);
    @(negedge clock);
    done.bits = '0;
    `ILF(done.bits, DONE, TAG) = 20;
    `ILF(done.bits, DONE, OK) = 1;
    query('h6000, 0, 0, 32);
    check(grants[21] == 0, "dependent RAW grant issued before final DMA ACK");
    finish_dma(20);
    query('h6000, 0, 0, 8);
    check(completions[20] == 0 && grants[21] == 0, "write retired before post-invalidate");
    @(negedge clock);
    complete.ready = 0;
    maintain(20, 2, 13);
    wait (complete.valid);
    query('h6000, 0, 0, 8);
    check(grants[21] == 0, "RAW grant overtook stalled predecessor completion");
    @(negedge clock);
    complete.ready = 1;
    await_complete(20);
    maintain(21, 0);
    await_grant(21);
    query('h6000, 0, 1);
    query('h6000, 1, 0);
    finish_dma(21);
    await_complete(21);
    query('h6000, 1, 1);
    // WAR then WAW enforce age while independent ranges retain forward progress.
    reserve(30);
    declare_access(30, 1, 'h7000, 64, 0);
    reserve(31);
    declare_access(31, 1, 'h7000, 64, 1);
    reserve(32);
    declare_access(32, 1, 'h7000, 64, 1);
    reserve(33);
    declare_access(33, 1, 'h8000, 64, 0);
    maintain();
    maintain();
    await_grant(30);
    await_grant(33);
    check(grants[31] == 0 && grants[32] == 0, "WAR/WAW bypassed older operation");
    query('h9000, 1, 1);
    finish_dma(33);
    await_complete(33);
    finish_dma(30);
    await_complete(30);
    maintain(31, 1);
    await_grant(31);
    check(grants[32] == 0, "WAW bypassed older writer");
    finish_dma(31);
    maintain(31, 2);
    await_complete(31);
    maintain(32, 1);
    await_grant(32);
    finish_dma(32);
    maintain(32, 2);
    await_complete(32);
    // Unknown older reservation blocks a younger known nonoverlapping memory access.
    reserve(40);
    reserve(41);
    declare_access(41, 1, 'hA000, 64, 0);
    query('hB000, 0, 0, 6);
    check(!maintenance.valid && grants[41] == 0, "younger access overtook unknown reservation");
    declare_access(40, 0, 0, 0, 0);
    await_complete(40);
    maintain(41, 0);
    await_grant(41);
    finish_dma(41);
    await_complete(41);
    // Reuse a fully retired ID at the highest legal PA line.
    reserve(255);
    declare_access(255, 1, (64'd1 << 44) - 64, 64, 1);
    query((64'd1 << 44) - 8, 0, 0);
    query((64'd1 << 44) - 72, 1, 1);
    maintain(255, 1);
    await_grant(255);
    finish_dma(255);
    maintain(255, 2);
    await_complete(255);
    query((64'd1 << 44) - 8, 0, 1);
    // Joint reset cancels an unknown reservation and a granted DMA operation externally.
    reserve(50);
    declare_access(50, 1, 'hB000, 64, 0);
    maintain(50, 0);
    await_grant(50);
    reserve(51);
    query('hC000, 0, 0);
    joint_reset();
    query('hB000, 1, 1);
    check(interlock_ref_live(model) == 0, "joint reset did not cancel outstanding operations");
    @(negedge clock);
    grant.ready = 1;
    complete.ready = 1;
    reserve(50);
    declare_access(50, 1, 'hB000, 64, 0);
    maintain(50, 0);
    await_grant(50);
    finish_dma(50);
    await_complete(50);
    // Reachable report holes: same-cycle dispatch+info holds info until its later handshake.
    @(negedge clock);
    dispatch.bits = '0;
    access_info.bits = '0;
    `ILF(dispatch.bits, DISPATCH, ID) = 254;
    dispatch.valid = 1;
    `ILF(access_info.bits, ACCESS_INFO, ID) = 254;
    `ILF(access_info.bits, ACCESS_INFO, LAST) = 1;
    access_info.valid = 1;
    cpu_query.bits = '0;
    `ILF(cpu_query.bits, CPU_QUERY, VALID) = 1;
    `ILF(cpu_query.bits, CPU_QUERY, PADDR) = 'h4000;
    `ILF(cpu_query.bits, CPU_QUERY, SIZELOG2) = 3;
    #1;
    check(!cpu_allow, "same-cycle info bypassed reservation hazard");
    @(posedge clock);
    check(dispatch.ready && !access_info.ready, "new info must await its reservation handshake");
    @(negedge clock);
    dispatch.valid = 0;
    do @(posedge clock); while (!access_info.ready);
    @(negedge clock);
    access_info.valid = 0;
    cpu_query.bits = '0;
    await_complete(254);
    // Each physical register slot sees complementary full-width IDs and high PA tags.
    for (int slot = 0; slot < 4; slot++) begin
      reserve(8'hf0 ^ slot);
      declare_access(8'hf0 ^ slot, 1, (64'd1 << 44) - 64, 64, 0);
    end
    for (int slot = 0; slot < 4; slot++) maintain();
    for (int slot = 0; slot < 4; slot++) await_grant(8'hf0 ^ slot);
    for (int slot = 3; slot >= 0; slot--) begin
      finish_dma(8'hf0 ^ slot);
      await_complete(8'hf0 ^ slot);
    end
    for (int slot = 0; slot < 4; slot++) begin
      reserve(8'h0f ^ slot);
      declare_access(8'h0f ^ slot, 1, 0, 64'd1 << 44, 1);
    end
    for (int slot = 0; slot < 4; slot++) begin
      maintain(8'h0f ^ slot, 1);
      await_grant(8'h0f ^ slot);
      finish_dma(8'h0f ^ slot);
      maintain(8'h0f ^ slot, 2);
      await_complete(8'h0f ^ slot);
    end
    // Extent input high bits are legal: no artificial small-range fixture restriction.
    for (int bit_index = 0; bit_index < 44; bit_index++) begin
      reserve(170);
      declare_access(170, 1, 0, 64'd1 << bit_index, 0);
      maintain(170, 0, 2);
      await_grant(170);
      finish_dma(170);
      await_complete(170);
    end
    // Sub-line extents are rounded to the cache-maintenance line ownership unit.
    reserve(171);
    declare_access(171, 1, 'h3003f, 1, 0);
    query('h30000, 1, 0);
    query('h30040, 1, 1);
    maintain(171, 0);
    await_grant(171);
    finish_dma(171);
    await_complete(171);
    // CPU byte/half/word accesses are legal and naturally aligned, up to one line.
    for (int size_log2 = 0; size_log2 <= 3; size_log2++)
    for (int offset = 0; offset < 8; offset += (1 << size_log2))
    query('h20000 + offset, 1, 1, 1, 0, size_log2);
    // Recycle low slots while higher older slots remain live; dependencies must not alias a new occupant.
    for (int slot = 0; slot < 4; slot++) begin
      reserve(100 + slot);
      declare_access(100 + slot, 1, 'hD000, 64, 0);
    end
    for (int slot = 0; slot < 4; slot++) maintain();
    for (int slot = 0; slot < 4; slot++) await_grant(100 + slot);
    for (int slot = 0; slot < 4; slot++) begin
      finish_dma(100 + slot);
      await_complete(100 + slot);
      reserve(200 + slot);
      declare_access(200 + slot, 1, 'hD000, 64, 1);
    end
    for (int slot = 0; slot < 4; slot++) begin
      maintain(200 + slot, 1);
      await_grant(200 + slot);
      finish_dma(200 + slot);
      maintain(200 + slot, 2);
      await_complete(200 + slot);
    end
    // Joint synchronous reset cancels legal in-flight producer offers at its first edge.
    reserve(60);
    declare_access(60, 1, 'hF000, 64, 0);
    maintain(60, 0);
    await_grant(60);
    reserve(61);
    declare_access(61, 1, 'h10000, 64, 1);
    wait (maintenance.valid);
    @(negedge clock);
    maintenance.ready = 1;
    @(posedge clock);
    @(negedge clock);
    maintenance.ready = 0;
    reserve(62);
    @(negedge clock);
    reset = 1;
    dispatch.bits = '0;
    `ILF(dispatch.bits, DISPATCH, ID) = 63;
    dispatch.valid = 1;
    access_info.bits = '0;
    `ILF(access_info.bits, ACCESS_INFO, ID) = 62;
    `ILF(access_info.bits, ACCESS_INFO, LAST) = 1;
    access_info.valid = 1;
    maintained.bits = '0;
    `ILF(maintained.bits, MAINTAINED, TAG) = 61;
    `ILF(maintained.bits, MAINTAINED, OK) = 1;
    maintained.valid = 1;
    done.bits = '0;
    `ILF(done.bits, DONE, TAG) = 60;
    `ILF(done.bits, DONE, OK) = 1;
    done.valid = 1;
    cpu_query.bits = '0;
    `ILF(cpu_query.bits, CPU_QUERY, VALID) = 1;
    `ILF(cpu_query.bits, CPU_QUERY, PADDR) = 'hF000;
    `ILF(cpu_query.bits, CPU_QUERY, SIZELOG2) = 3;
    @(posedge clock);
    @(negedge clock);
    dispatch.valid = 0;
    access_info.valid = 0;
    maintained.valid = 0;
    done.valid = 0;
    cpu_query.bits = '0;
    repeat (3) @(posedge clock);
    @(negedge clock);
    reset = 0;
    query(0, 1, 1);
    check(interlock_ref_live(model) == 0, "joint reset left producer-offer reservations live");
    // Fast external endpoints may offer ACK in the same cycle as request/grant acceptance.
    @(negedge clock);
    grant.ready = 0;
    reserve(0);
    declare_access(0, 1, 'hE000, 64, 0);
    wait (maintenance.valid);
    @(negedge clock);
    maintenance.ready = 1;
    maintained.bits = '0;
    `ILF(maintained.bits, MAINTAINED, TAG) = 0;
    `ILF(maintained.bits, MAINTAINED, OK) = 1;
    maintained.valid = 1;
    @(posedge clock);
    check(maintenance.valid && !maintained.ready,
          "fast cache response did not wait for accepted request");
    @(negedge clock);
    maintenance.ready = 0;
    do @(posedge clock); while (!maintained.ready);
    @(negedge clock);
    maintained.valid = 0;
    maintained.bits  = '0;
    wait (grant.valid);
    @(negedge clock);
    grant.ready = 1;
    done.bits = '0;
    `ILF(done.bits, DONE, TAG) = 0;
    `ILF(done.bits, DONE, OK) = 1;
    done.valid = 1;
    @(posedge clock);
    check(grant.valid && !done.ready, "early DMA completion did not wait for accepted grant");
    @(negedge clock);
    grant.ready = 0;
    do @(posedge clock); while (!done.ready);
    @(negedge clock);
    done.valid  = 0;
    done.bits   = '0;
    grant.ready = 1;
    await_complete(0);
    // An unsealed eight-extent list blocks all CPU queries; sealing exposes exact holes.
    reserve(80);
    for (int extent = 0; extent < 8; extent++) begin
      declare_access(80, 1, 'h200000 + extent * 128, 64, 1, extent == 7);
      if (extent != 7) begin
        query('h300000, 0, 0, 1);
        check(!maintenance.valid && grants[80] == 0,
              "partial unsealed list reached maintenance/grant");
      end
    end
    query('h200040, 1, 1);
    query('h200000, 0, 0);
    for (int extent = 0; extent < 8; extent++) begin
      maintain(80, 1, 2);
      check(grants[80] == 0 || extent == 7, "grant before every pre-maintenance ACK");
    end
    await_grant(80);
    finish_dma(80);
    for (int extent = 0; extent < 8; extent++) begin
      maintain(80, 2, 2);
      if (extent != 7) check(completions[80] == 0, "complete before every post-invalidate ACK");
    end
    await_complete(80);
    query('h200000, 1, 1);
    check(maintenance_ops[80][1] == 8 && maintenance_ops[80][2] == 8,
          "eight extents were not all maintained");
    // Preflight failure cancels unknown/partial reservations without producing a fake completion.
    reserve(81);
    query('h300000, 0, 0, 1);
    @(negedge clock);
    cancel.bits = '0;
    `ILF(cancel.bits, CANCEL, ID) = 81;
    cancel.valid = 1;
    do @(posedge clock); while (!cancel.ready);
    @(negedge clock);
    cancel.valid = 0;
    query('h300000, 0, 1, 1);
    check(completions[81] == 0 && grants[81] == 0, "empty cancel became grant/complete");
    reserve(82);
    declare_access(82, 1, 'h400000, 64, 0, 0);
    query('h500000, 0, 0, 1);
    @(negedge clock);
    cancel.bits = '0;
    `ILF(cancel.bits, CANCEL, ID) = 82;
    cancel.valid = 1;
    do @(posedge clock); while (!cancel.ready);
    @(negedge clock);
    cancel.valid = 0;
    query('h500000, 0, 1, 1);
    check(completions[82] == 0 && grants[82] == 0, "partial cancel became grant/complete");
    reserve(82);
    declare_access(82, 0, 0, 0, 0);
    await_complete(82);
    // Inactive payloads are unconstrained by valid: exercise arithmetic carry without an illegal handshake.
    @(negedge clock);
    access_info.valid = 0;
    access_info.bits = '1;
    cpu_query.bits = '0;
    `ILF(cpu_query.bits, CPU_QUERY, SIZELOG2) = 4;
    repeat (2) begin
      @(posedge clock);
      #1;
      check(cpu_allow, "inactive CPU query affected permit");
    end
    @(negedge clock);
    access_info.bits = '0;
    cpu_query.bits   = '0;
    repeat (2) @(posedge clock);
    check(interlock_ref_live(model) == 0, "normal test left unmatched transactions");
    `uvm_info("CHECKED", $sformatf(
              "queries=%0d reserved=%0d dmaACK=%0d completions=%0d",
              checked_queries,
              reserved,
              acknowledged,
              released
              ), UVM_LOW)
    interlock_ref_destroy(model);
    model_ready = 0;
    ctl.done = 1;
  end
endmodule
