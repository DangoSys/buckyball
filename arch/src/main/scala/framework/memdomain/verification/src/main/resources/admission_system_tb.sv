`include "admission_system_config.svh"
module admission_system_tb;
  import uvm_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle ddr_ref_create();
  import "DPI-C" function void admission_trace_init();
  import "DPI-C" function void dpi_bdb_set_clk(input longint unsigned cycle);
  import "DPI-C" function void ddr_ref_destroy(input chandle model);
  import "DPI-C" function void ddr_ref_program(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data
  );
  import "DPI-C" function int unsigned ddr_ref_read(
    input chandle model,
    input longint unsigned address,
    output bit [511:0] data
  );
  import "DPI-C" function int unsigned ddr_ref_write(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data,
    input longint unsigned mask
  );
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  ip_control_if ctl (clock);
  `include "admission_system_signals.svh"
AdmissionSystem dut (
      `include "admission_system_ports.svh"
  );
  `include "admission_system_clocking.svh"
stream_if #(64 + 64 + 8) command_if (
      clock,
      reset
  );
  assign command_if.valid = io_core_command_valid;
  assign command_if.ready = io_core_command_ready;
  assign command_if.bits = {
    io_core_command_bits_instruction_rs1Data,
    io_core_command_bits_instruction_rs2Data,
    io_core_command_bits_tag
  };
  stream_if #(8) complete_if (
      clock,
      reset
  );
  assign complete_if.valid = io_core_complete_valid;
  assign complete_if.ready = io_core_complete_ready;
  assign complete_if.bits  = io_core_complete_bits_tag;
  chandle model;
  bit returned[4] = '{default: 0};
  bit [511:0] line, wline;
  bit [63:0] wmask;
  int
      cycle = 0,
      completed = 0,
      reads = 0,
      writes = 0,
      backs = 0,
      aw_id = 0,
      ar_id = 0,
      rbeat = 0,
      rbeats = 0,
      wbeat = 0,
      wbeats = 0,
      r_due = 0,
      b_due = 0,
      maint_due = 0;
  bit reading = 0, writing = 0, committed = 0, roff = 0, boff = 0, maint_pending = 0, allow_b = 1;
  localparam int ROWS = `ADMISSION_ROWS;
  localparam int BYTES = ROWS * 16;
  localparam int EXPECTED_BURSTS = (ROWS + 255) / 256;
  localparam longint unsigned INPUT_BASE = 'h80010000;
  localparam longint unsigned OUTPUT_BASE = 'h80020000;
  localparam longint unsigned UNALIGNED_BASE = 'h80024ff3;
  int expected_total_b = EXPECTED_BURSTS, expected_store_tag = 2;
  bit unaligned_phase = 0, expectedFaultPhase = 0;
  int fault_start_aw = 0, fault_start_b = 0, fault_start_completed = 0;
  longint unsigned aw_address = 0, ar_address = 0;
  int maint_tag = 0;
  bit finish_wait = 0;
  function automatic bit [511:0] pattern(int line_index = 0);
    bit [511:0] data;
    for (int i = 0; i < 64; i++) data[i*8+:8] = line_index * 37 + i * 11 + 3;
    return data;
  endfunction
  task automatic verify_contract(bit ok, string message);
    if (!ok) `uvm_fatal("ADMISSION_SYSTEM", message)
  endtask
  task automatic memory_and_checks();
    forever begin
      @(sample);
      if (!sample.reset) begin
        cycle++;
        dpi_bdb_set_clk(cycle);
        if (!expectedFaultPhase)
          verify_contract(
              !sample.io_halted, $sformatf(
              "halted error%0d address%h", sample.io_fault_error, sample.io_fault_address));
        else begin
          verify_contract(!sample.io_core_complete_valid,
                          "errored store published Core completion");
          if (writes > fault_start_aw || sample.io_halted)
            verify_contract(!sample.io_core_cpuAllow, "errored write range was released to CPU");
          if (sample.io_halted)
            verify_contract(
                sample.io_fault_error == 2 && sample.io_fault_address == OUTPUT_BASE &&
                            sample.io_faultTag == 0,
                "late BERROR halted with incorrect typed fault or tag");
        end
        if (sample.io_core_complete_valid && sample.io_core_complete_ready) begin
          int id;
          id = sample.io_core_complete_bits_tag;
          verify_contract(!returned[id], "duplicate Core completion");
          returned[id] = 1;
          completed++;
          if (id == expected_store_tag)
            verify_contract(
                writes == expected_total_b &&
                            backs + int'(sample.io_axi_b_valid && sample.io_axi_b_ready) == expected_total_b &&
                            (!writing || (sample.io_axi_b_valid && sample.io_axi_b_ready)),
                "MVOUT Core release preceded final B for the whole bank");
        end
        if (sample.io_core_maintenance_valid && sample.io_core_maintenance_ready) begin
          verify_contract(!maint_pending, "maintenance overlap");
          maint_pending = 1;
          maint_tag = sample.io_core_maintenance_bits_tag;
          maint_due = cycle + 8;
        end
        if (sample.io_core_maintained_valid && sample.io_core_maintained_ready) maint_pending = 0;
        if (sample.io_axi_ar_valid && sample.io_axi_ar_ready) begin
          verify_contract(
              !reading && sample.io_axi_ar_bits_size == 4 && sample.io_axi_ar_bits_burst == 1 &&
              sample.io_axi_ar_bits_addr[3:0] == 0 && sample.io_axi_ar_bits_len + 1 <= 256 &&
              sample.io_axi_ar_bits_addr[11:0] + (sample.io_axi_ar_bits_len + 1) * 16 <= 4096,
              "wrong AR");
          verify_contract(
              sample.io_axi_ar_bits_addr == INPUT_BASE + reads * 4096 &&
                          sample.io_axi_ar_bits_len + 1 == ((ROWS - reads * 256) < 256 ? ROWS - reads * 256 : 256),
              "read did not fill a legal 256-beat/page segment");
          reading = 1;
          ar_id = sample.io_axi_ar_bits_id;
          rbeat = 0;
          rbeats = sample.io_axi_ar_bits_len + 1;
          ar_address = sample.io_axi_ar_bits_addr;
          r_due = cycle + 4;
          reads++;
        end
        if (sample.io_axi_r_valid && sample.io_axi_r_ready) begin
          roff = 0;
          rbeat++;
          r_due = cycle + 3;
          if (rbeat == rbeats) reading = 0;
        end
        if (sample.io_axi_aw_valid && sample.io_axi_aw_ready) begin
          verify_contract(
              !writing && sample.io_axi_aw_bits_size == 4 && sample.io_axi_aw_bits_burst == 1 &&
              sample.io_axi_aw_bits_addr[3:0] == 0 && sample.io_axi_aw_bits_len + 1 <= 256 &&
              sample.io_axi_aw_bits_addr[11:0] + (sample.io_axi_aw_bits_len + 1) * 16 <= 4096,
              "wrong AW");
          if (expectedFaultPhase)
            verify_contract(
                writes == fault_start_aw && sample.io_axi_aw_bits_addr == OUTPUT_BASE &&
                            sample.io_axi_aw_bits_len == 255,
                "BERROR case emitted a later page or lost its first 256-beat burst");
          else if (!unaligned_phase)
            verify_contract(
                sample.io_axi_aw_bits_addr == OUTPUT_BASE + writes * 4096 &&
                            sample.io_axi_aw_bits_len + 1 == ((ROWS - writes * 256) < 256 ? ROWS - writes * 256 : 256),
                "write did not fill a legal 256-beat/page segment");
          else begin
            int segment;
            segment = writes - EXPECTED_BURSTS;
            verify_contract(
                (segment == 0 && sample.io_axi_aw_bits_addr == (UNALIGNED_BASE & ~64'd15) && sample.io_axi_aw_bits_len == 0) ||
                            (segment == 1 && sample.io_axi_aw_bits_addr == ((UNALIGNED_BASE + 13) & ~64'd15) && sample.io_axi_aw_bits_len == 3),
                "unaligned cross-page AW segments wrong");
          end
          writing = 1;
          aw_id = sample.io_axi_aw_bits_id;
          aw_address = sample.io_axi_aw_bits_addr;
          wbeats = sample.io_axi_aw_bits_len + 1;
          wbeat = 0;
          wline = '0;
          wmask = '0;
          committed = 0;
          boff = 0;
          writes++;
        end
        if (sample.io_axi_w_valid && sample.io_axi_w_ready) begin
          verify_contract(writing && sample.io_axi_w_bits_last == (wbeat + 1 == wbeats),
                          "W lacks actual AW or WLAST incorrect");
          if (unaligned_phase)
            verify_contract(
                sample.io_axi_w_bits_strb == (writes == EXPECTED_BURSTS + 1 ? 'hfff8 : wbeat == wbeats - 1 ? 'h0007 : 'hffff),
                "unaligned cross-page first/tail WSTRB incorrect");
          begin
            longint unsigned beat_address;
            int line_offset;
            beat_address = aw_address + wbeat * 16;
            line_offset = beat_address & 63;
            wline = '0;
            wmask = '0;
            wline[line_offset*8+:128] = sample.io_axi_w_bits_data;
            wmask[line_offset+:16] = sample.io_axi_w_bits_strb;
            verify_contract(ddr_ref_write(model, beat_address & ~64'd63, wline, wmask) == 1,
                            "DDR beat commit failed across reference-line mapping");
          end
          wbeat++;
          if (wbeat == wbeats) begin
            committed = 1;
            b_due = cycle + 8;
          end
        end
        if (sample.io_axi_b_valid && sample.io_axi_b_ready) begin
          writing = 0;
          boff = 0;
          backs++;
        end
        if(sample.io_core_command_valid && sample.io_core_command_bits_instruction_opcode=='h2b && sample.io_core_command_bits_instruction_funct==2 &&
      (!returned[expected_store_tag] || backs < expected_total_b || writing))
          verify_contract(!sample.io_core_command_ready,
                          "Task finish published with pending DDR/ROB work");
      end
      @(negedge clock);
      io_axi_ar_ready = !reading && cycle % 5 != 0;
      io_axi_aw_ready = !writing && cycle % 4 != 0;
      io_axi_w_ready  = writing && wbeat < wbeats && cycle % 3 != 0;
      if (reading && !roff && cycle >= r_due) begin
        io_axi_r_valid   = 1;
        io_axi_r_bits_id = ar_id;
        begin
          longint unsigned beat_address;
          int line_offset;
          beat_address = ar_address + rbeat * 16;
          line_offset  = beat_address & 63;
          verify_contract(ddr_ref_read(model, beat_address & ~64'd63, line) == 1,
                          "unprogrammed DDR line during long read burst");
          io_axi_r_bits_data = line[line_offset*8+:128];
        end
        io_axi_r_bits_resp = 0;
        io_axi_r_bits_last = rbeat + 1 == rbeats;
        roff = 1;
      end
      if (!roff) io_axi_r_valid = 0;
      io_axi_b_valid = writing && committed && cycle >= b_due && allow_b;
      io_axi_b_bits_id = aw_id;
      io_axi_b_bits_resp = expectedFaultPhase ? 2 : 0;
      io_core_maintenance_ready = !maint_pending;
      io_core_maintained_valid = maint_pending && cycle >= maint_due;
      io_core_maintained_bits_tag = maint_tag;
      io_core_maintained_bits_ok = 1;
    end
  endtask
  task automatic controller(int fn, longint unsigned a, longint unsigned b,
                            output longint unsigned result);
    @(negedge clock);
    io_controller_cmd_bits_opcode = 'h2b;
    io_controller_cmd_bits_funct = fn;
    io_controller_cmd_bits_rs1Data = a;
    io_controller_cmd_bits_rs2Data = b;
    io_controller_cmd_bits_rd = 10;
    io_controller_cmd_valid = 1;
    io_controller_resp_ready = 1;
    do @(sample); while (!sample.io_controller_cmd_ready);
    @(negedge clock);
    io_controller_cmd_valid = 0;
    do @(sample); while (!sample.io_controller_resp_valid);
    result = sample.io_controller_resp_bits_data;
    @(negedge clock);
    io_controller_resp_ready = 0;
  endtask
  task automatic command(int tag, int opcode, int fn, longint unsigned a, longint unsigned b,
                         bit permission = 1);
    @(negedge clock);
    io_core_reserve_bits_id = tag;
    io_core_reserve_valid   = 1;
    do @(sample); while (!sample.io_core_reserve_ready);
    @(negedge clock);
    io_core_reserve_valid = 0;
    io_core_command_bits_tag = tag;
    io_core_command_bits_satp = 0;
    io_core_command_bits_effectivePrivilege = 1;
    io_core_command_bits_instruction_opcode = opcode;
    io_core_command_bits_instruction_funct = fn;
    io_core_command_bits_instruction_funct3 = 3;
    io_core_command_bits_instruction_xs1 = 1;
    io_core_command_bits_instruction_xs2 = 1;
    io_core_command_bits_instruction_rd = 10;
    io_core_command_bits_instruction_xd = opcode == 'h2b;
    io_core_command_bits_instruction_rs1Data = a;
    io_core_command_bits_instruction_rs2Data = b;
    io_core_command_bits_pmp_0_cfg_a = 1;
    io_core_command_bits_pmp_0_cfg_r = permission;
    io_core_command_bits_pmp_0_cfg_w = permission;
    io_core_command_bits_pmp_0_addr = 'h20400000;
    io_core_command_valid = 1;
    do @(sample); while (!sample.io_core_command_ready);
    @(negedge clock);
    io_core_command_valid = 0;
    // The live input is now denied. Permission must keep the accepted snapshot.
    io_core_command_bits_pmp_0_cfg_r = 0;
    io_core_command_bits_pmp_0_cfg_w = 0;
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 1ms);
    run_test("protocol_test");
  end
  initial begin
    longint unsigned value;
    bit [511:0] actual, first_expected, second_expected, source_data;
    `include "admission_system_init.svh"
admission_trace_init();
    model = ddr_ref_create();
    verify_contract(ROWS > 256 && BYTES % 64 == 0,
                    "fixture must exercise a full bank across pages");
    for (int index = 0; index < BYTES / 64; index++) begin
      ddr_ref_program(model, INPUT_BASE + index * 64, pattern(index));
      ddr_ref_program(model, OUTPUT_BASE + index * 64, '0);
    end
    wait (ctl.start);
    repeat (5) @(negedge clock);
    reset = 0;
    io_core_complete_ready = 1;
    io_core_response_ready = 1;
    fork
      memory_and_checks();
    join_none
    wait (io_npuIdle);
    command(3, 'h7b, 32, 0, 'h421, 0);
    wait (returned[3]);
    returned[3] = 0;
    for (int i = 0; i < 7; i++) controller(7, i, i == 6 ? `ADMISSION_SIGNATURE : 0, value);
    controller(0, 0, `ADMISSION_SIGNATURE, value);
    verify_contract(value == 0, "actual signature task submit failed");
    command(3, 'h2b, 8, 0, 0);
    wait (returned[3]);
    returned[3] = 0;
    io_blockedResponse = 1;
    io_core_cpuQuery_valid = 1;
    io_core_cpuQuery_paddr = INPUT_BASE;
    io_core_cpuQuery_sizeLog2 = 3;
    io_core_cpuQuery_write = 1;
    command(0, 'h7b, 33, 64'(ROWS) << 30, INPUT_BASE | (64'd1 << 39));
    @(sample);
    verify_contract(!sample.io_core_cpuAllow, "CPU write passed reserved NPU read range");
    repeat (12) @(sample);
    verify_contract(!returned[0], "context released at request acceptance");
    @(negedge clock);
    io_blockedResponse = 0;
    wait (returned[0]);
    @(sample);
    verify_contract(sample.io_core_cpuAllow, "CPU remained blocked after read completion");
    @(negedge clock);
    allow_b = 0;
    io_core_cpuQuery_paddr = OUTPUT_BASE;
    io_core_cpuQuery_write = 0;
    command(2, 'h7b, 16, 64'(ROWS) << 30, OUTPUT_BASE | (64'd1 << 39));
    wait (committed);
    @(sample);
    verify_contract(!sample.io_core_cpuAllow,
                    "CPU read passed NPU write before real DDR B and post-maintenance");
    fork
      command(3, 'h2b, 2, 0, 0, 0);
      begin
        finish_wait = 1;
        repeat (12) @(sample);
        verify_contract(!returned[2], "MVOUT released before real B");
        @(negedge clock);
        finish_wait = 0;
        allow_b = 1;
      end
    join
    wait (returned[2] && returned[3] && io_workDrained);
    @(sample);
    verify_contract(sample.io_core_cpuAllow,
                    "CPU read did not resume after actual write completion");
    verify_contract(
        reads == EXPECTED_BURSTS && writes == EXPECTED_BURSTS &&
                    backs == writes && !writing && !reading,
        "full-bank AXI segments not drained");
    for (int index = 0; index < BYTES / 64; index++)
    verify_contract(ddr_ref_read(model, OUTPUT_BASE + index * 64, actual) == 1 && actual == pattern(
                    index), "full-bank MVIN/MVOUT DDR golden mismatch");
    controller(1, 0, 0, value);
    verify_contract(value == 1, "Task finish failed to publish actual completion");
    // Reuse the released tag for a tiny real WriteDma carry/mask transaction.
    ddr_ref_program(model, UNALIGNED_BASE & ~64'd63, {64{8'ha5}});
    ddr_ref_program(model, (UNALIGNED_BASE + 13) & ~64'd63, {64{8'ha5}});
    @(negedge clock);
    returned[0] = 0;
    expected_store_tag = 0;
    expected_total_b = EXPECTED_BURSTS + 2;
    unaligned_phase = 1;
    allow_b = 0;
    committed = 0;
    io_core_cpuQuery_paddr = UNALIGNED_BASE & ~64'd7;
    io_core_cpuQuery_write = 0;
    command(0, 'h7b, 16, 64'd4 << 30, UNALIGNED_BASE | (64'd1 << 39));
    wait (committed);
    repeat (12) begin
      @(sample);
      verify_contract(!returned[0] && !sample.io_core_cpuAllow,
                      "unaligned store released before its delayed first B");
    end
    @(negedge clock);
    allow_b = 1;
    wait (returned[0] && io_workDrained);
    @(sample);
    verify_contract(
        backs == expected_total_b && writes == expected_total_b && !writing && !reading &&
                    sample.io_core_cpuAllow,
        "unaligned Core release did not join both final B responses");
    source_data = pattern(0);
    first_expected = {64{8'ha5}};
    second_expected = {64{8'ha5}};
    first_expected[51*8+:13*8] = source_data[0+:13*8];
    second_expected[0+:51*8] = source_data[13*8+:51*8];
    verify_contract(ddr_ref_read(model, UNALIGNED_BASE & ~64'd63, actual
                    ) == 1 && actual == first_expected,
                    "unaligned head carry corrupted requested data or three sentinel bytes");
    verify_contract(ddr_ref_read(model, (UNALIGNED_BASE + 13) & ~64'd63, actual
                    ) == 1 && actual == second_expected,
                    "cross-page carry/tail corrupted requested data or thirteen sentinel bytes");
    // First-page SLVERR is terminal: drain the remaining raw input without
    // issuing later AWs, and retain the Admission reservation instead of release.
    @(negedge clock);
    fault_start_aw = writes;
    fault_start_b = backs;
    fault_start_completed = completed;
    expectedFaultPhase = 1;
    unaligned_phase = 0;
    allow_b = 1;
    committed = 0;
    returned[0] = 0;
    io_core_cpuQuery_paddr = OUTPUT_BASE;
    io_core_cpuQuery_write = 0;
    command(0, 'h7b, 16, 64'(ROWS) << 30, OUTPUT_BASE | (64'd1 << 39));
    do @(sample); while (!(sample.io_halted && sample.io_npuIdle));
    repeat (24) begin
      @(sample);
      verify_contract(
          sample.io_halted && sample.io_npuIdle && sample.io_fault_error == 2 &&
                      sample.io_fault_address == OUTPUT_BASE && sample.io_faultTag == 0,
          $sformatf(
          "BERROR terminal error=%0d addr=%h tag=%0d idle=%0d halted=%0d expected=%h",
          sample.io_fault_error,
          sample.io_fault_address,
          sample.io_faultTag,
          sample.io_npuIdle,
          sample.io_halted,
          OUTPUT_BASE
          ));
      verify_contract(
          writes == fault_start_aw + 1 && backs == fault_start_b + 1 &&
                      !writing && !reading && !sample.io_axi_aw_valid,
          "BERROR emitted an extra AW/B or left an AXI transaction live");
      verify_contract(
          !returned[0] && completed == fault_start_completed &&
                      !sample.io_core_complete_valid && !sample.io_core_cpuAllow && !sample.io_workDrained,
          "BERROR released its Core/CPU reservation instead of retaining the failed context");
    end
    `uvm_info("ADMISSION_BERROR",
              "First 256-beat B=SLVERR: remaining input drained, one AW/B only, fault tag/address retained and no Core/CPU release",
              UVM_LOW)
    `uvm_info("ADMISSION_SYSTEM", $sformatf(
              "Completed%0d AR%0d AW%0d B%0d; actual Goban signature/PMP frozen, Task finish joined",
              completed,
              reads,
              writes,
              backs
              ), UVM_LOW)
    ddr_ref_destroy(model);
    ctl.done = 1;
  end
endmodule
