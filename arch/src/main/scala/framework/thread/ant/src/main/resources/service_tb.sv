`include "config.svh"
module service_tb;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle ant_hart_create(
    int unsigned code_bytes,
    longint unsigned tls_base,
    int unsigned tls_bytes,
    longint unsigned tss_base,
    int unsigned tss_bytes
  );
  import "DPI-C" function void ant_hart_destroy(chandle hart);
  import "DPI-C" function void ant_hart_code(
    chandle hart,
    int unsigned offset,
    int unsigned word
  );
  import "DPI-C" function void ant_hart_store(
    chandle hart,
    longint unsigned address,
    longint unsigned value
  );
  import "DPI-C" function longint unsigned ant_hart_load(
    chandle hart,
    longint unsigned address
  );
  import "DPI-C" function longint unsigned ant_hart_register(
    chandle hart,
    int unsigned index
  );
  import "DPI-C" function void ant_hart_start(
    chandle hart,
    longint unsigned entry,
    argument,
    stack
  );
  import "DPI-C" function int unsigned ant_hart_step(
    chandle hart,
    longint unsigned cycle,
    output longint unsigned pc,
    output int unsigned instruction,
    output int unsigned rd,
    output longint unsigned data
  );
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  scenario_if control (clock);
  ant_control_if management (
      clock,
      reset
  );
  logic [1:0] drained = '1, command_valid, command_ready = '1, retired_valid;
  logic [31:0] command_instruction[2];
  logic [63:0] command_a[2], command_b[2], command_pc[2], command_satp[2];
  logic [1:0] command_privilege[2], command_sum, command_mxr;
  logic [`PMP_BITS-1:0] command_pmp[2];
  logic [63:0] retired_pc[2], retired_data[2];
  logic [31:0] retired_instruction[2];
  logic [4:0] retired_rd[2];
  logic [4:0] response_rd[2] = '{0, 0};
  logic [63:0] response_data[2] = '{0, 0};
  // Random submitter contexts must not become thread accelerator address-space state.
  logic [63:0] request_satp = 0, task_satp[2];
  logic [1:0] request_privilege = 0, task_privilege[2];
  logic request_sum = 0, request_mxr = 0;
  logic [1:0] task_sum, task_mxr;
  logic [`PMP_BITS-1:0] request_pmp = 0, task_pmp[2];
  int completed = 0;
  chandle harts[2];
  bit [1:0] checking = 0, chaos = 0;
  longint unsigned cycles, retirements = 0, commands = 0;
  semaphore lock = new(1);
  ServiceVerification dut (
      .clock(clock),
      .reset(reset),
      .io_rocc_0_cmd_valid(command_valid[0]),
      .io_rocc_0_cmd_ready(command_ready[0]),
      .io_rocc_0_cmd_bits_instruction_raw_inst(command_instruction[0]),
      .io_rocc_0_cmd_bits_instruction_pc(command_pc[0]),
      .io_rocc_0_cmd_bits_instruction_rs1Data(command_a[0]),
      .io_rocc_0_cmd_bits_instruction_rs2Data(command_b[0]),
      .io_rocc_0_cmd_bits_satp(command_satp[0]),
      .io_rocc_0_cmd_bits_effectivePrivilege(command_privilege[0]),
      .io_rocc_0_cmd_bits_sum(command_sum[0]),
      .io_rocc_0_cmd_bits_mxr(command_mxr[0]),
      .io_rocc_0_resp_ready(),
      .io_rocc_0_busy(!drained[0]),
      .io_rocc_0_interrupt(1'b0),
      .io_rocc_0_resp_valid(1'b0),
      .io_rocc_0_resp_bits_rd(response_rd[0]),
      .io_rocc_0_resp_bits_data(response_data[0]),
      .io_retired_0_valid(retired_valid[0]),
      .io_retired_0_bits_pc(retired_pc[0]),
      .io_retired_0_bits_instruction(retired_instruction[0]),
      .io_retired_0_bits_rd(retired_rd[0]),
      .io_retired_0_bits_data(retired_data[0]),
      .io_rocc_1_cmd_valid(command_valid[1]),
      .io_rocc_1_cmd_ready(command_ready[1]),
      .io_rocc_1_cmd_bits_instruction_raw_inst(command_instruction[1]),
      .io_rocc_1_cmd_bits_instruction_pc(command_pc[1]),
      .io_rocc_1_cmd_bits_instruction_rs1Data(command_a[1]),
      .io_rocc_1_cmd_bits_instruction_rs2Data(command_b[1]),
      .io_rocc_1_cmd_bits_satp(command_satp[1]),
      .io_rocc_1_cmd_bits_effectivePrivilege(command_privilege[1]),
      .io_rocc_1_cmd_bits_sum(command_sum[1]),
      .io_rocc_1_cmd_bits_mxr(command_mxr[1]),
      .io_rocc_1_resp_ready(),
      .io_rocc_1_busy(!drained[1]),
      .io_rocc_1_interrupt(1'b0),
      .io_rocc_1_resp_valid(1'b0),
      .io_rocc_1_resp_bits_rd(response_rd[1]),
      .io_rocc_1_resp_bits_data(response_data[1]),
      .io_retired_1_valid(retired_valid[1]),
      .io_retired_1_bits_pc(retired_pc[1]),
      .io_retired_1_bits_instruction(retired_instruction[1]),
      .io_retired_1_bits_rd(retired_rd[1]),
      .io_retired_1_bits_data(retired_data[1]),
      .io_request_valid(management.valid),
      .io_request_ready(management.ready),
      .io_request_bits_control_operation(management.operation),
      .io_request_bits_control_context(management.context_id),
      .io_request_bits_control_field(management.field),
      .io_request_bits_control_data(management.data),
      .io_request_bits_context_satp(request_satp),
      .io_request_bits_context_effectivePrivilege(request_privilege),
      .io_request_bits_context_sum(request_sum),
      .io_request_bits_context_mxr(request_mxr),
      .io_reply_valid(management.reply_valid),
      .io_reply_ready(management.reply_ready),
      .io_reply_bits(management.reply)
      `include "service_pmp.svh"
  );
  // mcycle counts every cycle since reset; a `csrr mcycle` retires the pre-edge count.
  always @(posedge clock) cycles <= reset ? 0 : cycles + 1;
  always @(posedge clock)
    if (!reset)
      for (int c = 0; c < 2; c++) begin
        if (command_valid[c] && command_ready[c] && !retired_valid[c])
          `uvm_fatal("COMMAND", $sformatf("context %0d: command accepted without retirement", c))
        if (retired_valid[c] && checking[c]) begin
          longint unsigned pc, data, a, b;
          int unsigned instruction, rd, kind;
          a = ant_hart_register(harts[c], retired_instruction[c][19:15]);
          b = ant_hart_register(harts[c], retired_instruction[c][24:20]);
          kind = ant_hart_step(harts[c], cycles, pc, instruction, rd, data);
          if (kind == 1 || retired_pc[c] !== pc || retired_instruction[c] !== instruction ||
              retired_rd[c] !== rd || retired_data[c] !== data)
            `uvm_fatal("RETIRE", $sformatf(
                       "context %0d: pc %h insn %h rd %0d data %h, reference pc %h insn %h rd %0d data %h",
                       c,
                       retired_pc[c],
                       retired_instruction[c],
                       retired_rd[c],
                       retired_data[c],
                       pc,
                       instruction,
                       rd,
                       data
                       ))
          if ((kind == 2) !== (command_valid[c] && command_ready[c]))
            `uvm_fatal("COMMAND", $sformatf("context %0d: custom retirement and command differ", c))
          if (kind == 2) begin
            if (command_instruction[c] !== instruction || command_pc[c] !== pc ||
                command_a[c] !== (instruction[13] ? a : 0) ||
                command_b[c] !== (instruction[12] ? b : 0))
              `uvm_fatal("COMMAND", $sformatf("context %0d: command %h differs", c, instruction))
            if (command_satp[c] !== 0 || command_privilege[c] !== 3 ||
                command_sum[c] !== 0 || command_mxr[c] !== 0 || command_pmp[c] !== 0)
              `uvm_fatal("CONTEXT", $sformatf(
                         "context %0d: thread command carries address-space state", c))
            commands++;
          end
          retirements++;
          evidence::comparisons++;
          evidence::expected++;
          evidence::received++;
        end
      end
  always @(negedge clock)
    for (int c = 0; c < 2; c++)
      if (chaos[c]) begin
        drained[c] = $urandom_range(0, 3) != 0;
        command_ready[c] = $urandom_range(0, 2) != 0;
        response_rd[c] = $urandom;
        response_data[c] = {$urandom, $urandom};
      end
  initial begin
    #0;
    foreach (harts[c])
    harts[c] = ant_hart_create(`CODE_BYTES, `TLS_BASE, `TLS_BYTES, `TSS_BASE, `TSS_BYTES);
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end
  final foreach (harts[c]) ant_hart_destroy(harts[c]);
  // Management requests from concurrent context threads are serialized; every request
  // carries a fresh random submitter context.
  task automatic request(int op, int ctx, int field, longint unsigned value,
                         output longint unsigned reply);
    lock.get();
    request_satp = {$urandom, $urandom};
    request_privilege = $urandom;
    request_sum = $urandom;
    request_mxr = $urandom;
    for (int i = 0; i < `PMP_BITS; i += 32) request_pmp[i+:32] = $urandom;
    if (op == 2) begin
      task_satp[ctx] = request_satp;
      task_privilege[ctx] = request_privilege;
      task_sum[ctx] = request_sum;
      task_mxr[ctx] = request_mxr;
      task_pmp[ctx] = request_pmp;
    end
    management.transfer(op, ctx, field, value, reply);
    lock.put();
  endtask
  task automatic command(int op, int ctx, int field, longint unsigned value = 0);
    longint unsigned reply;
    request(op, ctx, field, value, reply);
    evidence::check(reply === 0, $sformatf("op%0d ctx%0d: reply %h", op, ctx, reply));
  endtask
  task automatic put(int port, longint unsigned address, logic [127:0] data);
    longint unsigned result, offset, shared_offset, descriptor;
    int space, ctx;
    space = port == 4 ? 2 : port % 2;
    ctx = port == 4 ? 0 : port / 2;
    offset = address - (space == 0 ? 0 : space == 1 ? `TLS_BASE : `TSS_BASE);
    shared_offset = space == 2 ? offset : `TSS_SCRATCH - `TSS_BASE;
    for (int half = 0; half < 2; half++)
      command(12, 0, shared_offset + 8 * half, data[64*half+:64]);
    if (space != 2) begin
      descriptor = (64'd16 << 32) | shared_offset;
      command(8 + 2 * space, ctx, offset, descriptor);
      command(7 + 2 * space, ctx, offset, descriptor);
    end
    for (int half = 0; half < 2; half++) begin
      request(11, 0, shared_offset + 8 * half, 0, result);
      if (result !== data[64*half+:64]) `uvm_fatal("LOAD", "TSS transfer readback mismatch")
    end
  endtask
  task automatic launch(int ctx, int task_id);
    command(1, ctx, 0, task_id);
    command(1, ctx, 1, 0);
    command(1, ctx, 2, 8);
    command(1, ctx, 3, `TLS_BASE);
    command(1, ctx, 4, `TLS_END);
    command(1, ctx, 5, ctx ? `SIGNATURE_1 : `SIGNATURE_0);
    command(2, ctx, 0);
  endtask
  task automatic finish(int ctx, int task_id, longint unsigned value, bit cancelled = 0);
    longint unsigned reply, status;
    do request(0, ctx, 1, 0, status); while (!(status & 4));
    if (status !== (cancelled ? 13 : 5)) `uvm_fatal("RESULT", "wrong terminal state")
    request(0, ctx, 8, 0, reply);
    if (reply !== task_id) `uvm_fatal("RESULT", "wrong task ID")
    request(0, ctx, 9, 0, reply);
    if (reply !== value)
      `uvm_fatal("RESULT", $sformatf("terminal value %h expected %h", reply, value))
    command(3, ctx, 0);
    completed++;
    evidence::checks++;
  endtask
  `include "ant_generator.svh"
  `include "service_programs.svh"
  `include "service_reset.svh"
  `include "service_cancel_memory.svh"
  initial begin
    longint unsigned reply;
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    put(1, `TLS_BASE, 128'h1111);
    put(3, `TLS_BASE, 128'h2222);
    put(4, `TSS_BASE, 128'h3333);
    put(0, 0, {64'b0, 32'h00000073, 32'h02500513});
    put(2, 0, {64'b0, 32'h00000073, 32'h01d00513});
    command(5, 0, 0);
    launch(0, 1);
    launch(1, 2);
    finish(0, 1, 37);
    finish(1, 2, 29);
    command(6, 0, 0);
    put(0, 0, {64'b0, 32'h00000073, 32'h0000307b});
    command_ready[0] = 0;
    command(5, 0, 0);
    launch(0, 3);
    drained[0] = 0;
    wait (command_valid[0]);
    command(4, 0, 0);
    request(0, 0, 1, 0, reply);
    if (reply !== 3) `uvm_fatal("DRAIN", "task completed before external work drained")
    drained[0] = 1;
    finish(0, 3, 0, 1);
    command(6, 0, 0);
    // A new task must not inherit the old task's cancellation.
    command_ready[0] = 1;
    put(0, 0, {64'b0, 32'h00000073, 32'h00900513});
    command(5, 0, 0);
    launch(0, 4);
    finish(0, 4, 9);
    command(6, 0, 0);
    // Cancellation remains effective after ECALL while backend work is still pending.
    command(5, 0, 0);
    launch(0, 5);
    drained[0] = 0;
    repeat (30) @(negedge clock);
    command(4, 0, 0);
    drained[0] = 1;
    finish(0, 5, 0, 1);
    command(6, 0, 0);
    cancel_memory();
    command(5, 0, 0);
    fork
      programs(0, 60);
      programs(1, 60);
    join
    command(6, 0, 0);
    service_reset();
    `uvm_info("SERVICE", $sformatf(
              "%0d real Ant tasks; %0d control transactions; %0d retirements (%0d commands) checked",
              completed,
              management.checked,
              retirements,
              commands
              ), UVM_LOW)
    control.done = 1;
  end
endmodule
