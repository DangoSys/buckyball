`include "config.svh"
module service_tb;
  import uvm_pkg::*;
  import ip_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  ip_control_if control (clock);
  ant_control_if management (
      clock,
      reset
  );
  logic [1:0] drained = '1, command_valid, command_ready = '1, response_valid = 0, response_ready;
  logic [31:0] command_instruction[2], command_task[2], response_task[2] = '{0, 0};
  logic [63:0] command_a[2], command_b[2], response_data[2] = '{0, 0}, retired_pc[2];
  logic [4:0] response_rd[2] = '{0, 0};
  logic [1:0] response_error = 0, retired_valid, cancel_npu;
  int completed = 0;
  ServiceVerification dut (
      .clock(clock),
      .reset(reset),
      .io_npuDrained_0(drained[0]),
      .io_command_0_ready(command_ready[0]),
      .io_command_0_valid(command_valid[0]),
      .io_command_0_bits_instruction(command_instruction[0]),
      .io_command_0_bits_task(command_task[0]),
      .io_command_0_bits_rs1(command_a[0]),
      .io_command_0_bits_rs2(command_b[0]),
      .io_retired_0_valid(retired_valid[0]),
      .io_retired_0_bits_pc(retired_pc[0]),
      .io_response_0_valid(response_valid[0]),
      .io_response_0_ready(response_ready[0]),
      .io_response_0_bits_task(response_task[0]),
      .io_response_0_bits_rd(response_rd[0]),
      .io_response_0_bits_data(response_data[0]),
      .io_response_0_bits_error(response_error[0]),
      .io_npuDrained_1(drained[1]),
      .io_command_1_ready(command_ready[1]),
      .io_command_1_valid(command_valid[1]),
      .io_command_1_bits_instruction(command_instruction[1]),
      .io_command_1_bits_task(command_task[1]),
      .io_command_1_bits_rs1(command_a[1]),
      .io_command_1_bits_rs2(command_b[1]),
      .io_retired_1_valid(retired_valid[1]),
      .io_retired_1_bits_pc(retired_pc[1]),
      .io_response_1_valid(response_valid[1]),
      .io_response_1_ready(response_ready[1]),
      .io_response_1_bits_task(response_task[1]),
      .io_response_1_bits_rd(response_rd[1]),
      .io_response_1_bits_data(response_data[1]),
      .io_response_1_bits_error(response_error[1]),
      .io_request_valid(management.valid),
      .io_request_ready(management.ready),
      .io_request_bits_operation(management.operation),
      .io_request_bits_context(management.context_id),
      .io_request_bits_field(management.field),
      .io_request_bits_data(management.data),
      .io_reply_valid(management.reply_valid),
      .io_reply_ready(management.reply_ready),
      .io_reply_bits(management.reply),
      .io_signatures_0(64'h1234),
      .io_online_0(1'b1),
      .io_cancelNpu_0(cancel_npu[0]),
      .io_signatures_1(64'h1235),
      .io_online_1(1'b1),
      .io_cancelNpu_1(cancel_npu[1])
  );
  initial begin
    #0;
    uvm_config_db#(virtual ip_control_if)::set(null, "uvm_test_top", "vif", control);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 100us);
    run_test("protocol_test");
  end
  task automatic command(int op, int ctx, int field, longint unsigned value = 0);
    longint unsigned reply;
    management.transfer(op, ctx, field, value, reply);
  endtask
  task automatic put(int port, longint unsigned address, logic [127:0] data);
    longint unsigned result, offset;
    int space, ctx;
    space = port == 4 ? 2 : port % 2;
    ctx = port == 4 ? 0 : port / 2;
    offset = address - (space == 0 ? 0 : space == 1 ? `TLS_BASE : `TSS_BASE);
    for (int half = 0; half < 2; half++) begin
      command(8 + 2 * space, ctx, offset + 8 * half, data[64*half+:64]);
      management.transfer(7 + 2 * space, ctx, offset + 8 * half, 0, result);
      if (result !== data[64*half+:64]) `uvm_fatal("LOAD", "local storage readback mismatch")
    end
  endtask
  task automatic launch(int ctx, int task_id);
    command(1, ctx, 0, task_id);
    command(1, ctx, 1, 0);
    command(1, ctx, 2, 8);
    command(1, ctx, 3, `TLS_BASE);
    command(1, ctx, 4, `TLS_END);
    command(1, ctx, 5, 64'h1234 + ctx);
    command(2, ctx, 0);
  endtask
  task automatic finish(int ctx, int task_id, longint unsigned value, bit cancelled = 0);
    longint unsigned reply, status;
    do management.transfer(0, ctx, 1, 0, status); while (!(status & 4));
    if (status !== (cancelled ? 13 : 5)) `uvm_fatal("RESULT", "wrong terminal state")
    management.transfer(0, ctx, 8, 0, reply);
    if (reply !== task_id) `uvm_fatal("RESULT", "wrong task ID")
    management.transfer(0, ctx, 9, 0, reply);
    if (reply !== value) `uvm_fatal("RESULT", "wrong terminal value")
    command(3, ctx, 0);
    completed++;
  endtask
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
    put(0, 0, {64'b0, 32'h00000073, 32'h0000007b});
    command_ready[0] = 0;
    command(5, 0, 0);
    launch(0, 3);
    drained[0] = 0;
    wait (command_valid[0]);
    command(4, 0, 0);
    management.transfer(0, 0, 1, 0, reply);
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
    `uvm_info("SERVICE", $sformatf(
              "%0d real Ant tasks; %0d control transactions checked", completed, management.checked
              ), UVM_LOW)
    control.done = 1;
  end
endmodule
