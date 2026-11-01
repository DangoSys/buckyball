`include "config.svh"
module control_tb;
  import uvm_pkg::*;
  import scenario_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  scenario_if control (clock);
  ant_control_if management (
      clock,
      reset
  );
  logic [1:0] start_valid, start_ready = '1, result_valid = 0, result_ready, cancel, available;
  logic storage_busy = 0, in_use;
  logic [31:0] task_id[2], result_task[2] = '{7, 8};
  logic [63:0] result_value[2] = '{64'h42, 64'h42};
  logic [1:0] result_cancelled = 0;
  logic [63:0] entry[2], code_end[2], argument[2], stack[2];
  logic [63:0] signature[2];
  // The submitter's context accompanies every request; Start forwards the accepted one.
  logic [63:0] request_satp = 64'h8000000000000bad, launch_satp[2];
  logic [1:0] request_privilege = 0, launch_privilege[2];
  logic request_sum = 0, request_mxr = 0;
  logic [1:0] launch_sum, launch_mxr;
  logic [`PMP_BITS-1:0] request_pmp = 0, launch_pmp[2];
  // Reference task registers: staged descriptors, ownership and completion.
  logic [63:0] staged[2][5] = '{'{7, 0, 16, `TLS_BASE, `TLS_END}, '{8, 0, 16, `TLS_BASE, `TLS_END}};
  bit owned[2], done[2];
  int checked = 0, launches = 0, cancels = 0;
  Control dut (
      .clock(clock),
      .reset(reset),
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
      .io_response_valid(management.reply_valid),
      .io_response_ready(management.reply_ready),
      .io_response_bits(management.reply),
      .io_storageBusy(storage_busy),
      .io_inUse(in_use),
      .io_available_0(available[0]),
      .io_available_1(available[1]),
      .io_cancel_0(cancel[0]),
      .io_start_0_valid(start_valid[0]),
      .io_start_0_ready(start_ready[0]),
      .io_start_0_bits_descriptor_task(task_id[0]),
      .io_start_0_bits_descriptor_entry(entry[0]),
      .io_start_0_bits_descriptor_codeEnd(code_end[0]),
      .io_start_0_bits_descriptor_argument(argument[0]),
      .io_start_0_bits_descriptor_stack(stack[0]),
      .io_start_0_bits_context_satp(launch_satp[0]),
      .io_start_0_bits_context_effectivePrivilege(launch_privilege[0]),
      .io_start_0_bits_context_sum(launch_sum[0]),
      .io_start_0_bits_context_mxr(launch_mxr[0]),
      .io_result_0_valid(result_valid[0]),
      .io_result_0_ready(result_ready[0]),
      .io_result_0_bits_task(result_task[0]),
      .io_result_0_bits_cancelled(result_cancelled[0]),
      .io_result_0_bits_value(result_value[0]),
      .io_cancel_1(cancel[1]),
      .io_start_1_valid(start_valid[1]),
      .io_start_1_ready(start_ready[1]),
      .io_start_1_bits_descriptor_task(task_id[1]),
      .io_start_1_bits_descriptor_entry(entry[1]),
      .io_start_1_bits_descriptor_codeEnd(code_end[1]),
      .io_start_1_bits_descriptor_argument(argument[1]),
      .io_start_1_bits_descriptor_stack(stack[1]),
      .io_start_1_bits_context_satp(launch_satp[1]),
      .io_start_1_bits_context_effectivePrivilege(launch_privilege[1]),
      .io_start_1_bits_context_sum(launch_sum[1]),
      .io_start_1_bits_context_mxr(launch_mxr[1]),
      .io_result_1_valid(result_valid[1]),
      .io_result_1_ready(result_ready[1]),
      .io_result_1_bits_task(result_task[1]),
      .io_result_1_bits_cancelled(result_cancelled[1]),
      .io_result_1_bits_value(result_value[1])
      `include "control_pmp.svh"
  );
  always @(posedge clock)
    if (!reset) begin
      for (int i = 0; i < 2; i++) begin
        if (start_valid[i] && start_ready[i]) begin
          if (task_id[i] !== staged[i][0][31:0] || entry[i] !== staged[i][1] ||
              code_end[i] !== staged[i][2] || argument[i] !== staged[i][3] ||
              stack[i] !== staged[i][4])
            `uvm_fatal("START", $sformatf("context %0d: launch descriptor differs from staging", i))
          if (launch_satp[i] !== request_satp || launch_privilege[i] !== request_privilege ||
              launch_sum[i] !== request_sum || launch_mxr[i] !== request_mxr ||
              launch_pmp[i] !== request_pmp)
            `uvm_fatal("CONTEXT", "launch context does not belong to the accepted start command")
          launches++;
        end
        if (cancel[i] && !(management.valid && management.operation == 4 &&
                           management.context_id == i && owned[i]))
          `uvm_fatal("CANCEL", $sformatf("context %0d: cancel without a cancel request", i))
        if (cancel[i]) cancels++;
        if (available[i] !== (!owned[i] && !done[i]))
          `uvm_fatal("AVAILABLE", $sformatf("context %0d availability differs", i))
        if (result_ready[i] !== (owned[i] && !done[i]))
          `uvm_fatal("RESULT", $sformatf("context %0d result readiness differs", i))
        if (result_valid[i] && result_ready[i]) begin
          owned[i] = 0;
          done[i]  = 1;
        end
      end
      if (management.valid && management.ready) begin
        if (management.operation == 4 && owned[management.context_id] &&
            !cancel[management.context_id])
          `uvm_fatal("CANCEL", "cancel request for a live task did not reach the thread")
        if (management.operation == 2) owned[management.context_id] = 1;
        if (management.operation == 3) done[management.context_id] = 0;
        if (management.operation == 1 && management.field < 5)
          staged[management.context_id][management.field] = management.data;
      end
    end
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end
  task automatic transact(int op, int ctx, int f, longint unsigned value,
                          longint unsigned expected = 0);
    longint unsigned actual;
    management.transfer(op, ctx, f, value, actual);
    verification_pkg::evidence::check(
        actual === expected, $sformatf(
        "op%0d ctx%0d field%0d: data %h expected %h", op, ctx, f, actual, expected));
    checked++;
  endtask
  task automatic descriptor(int ctx);
    transact(1, ctx, 0, 7 + ctx);
    transact(1, ctx, 1, 0);
    transact(1, ctx, 2, 16);
    transact(1, ctx, 3, `TLS_BASE);
    transact(1, ctx, 4, `TLS_END);
    transact(1, ctx, 5, signature[ctx]);
  endtask
  task automatic complete(int ctx, longint unsigned value, bit was_cancelled);
    @(negedge clock);
    result_task[ctx] = staged[ctx][0];
    result_value[ctx] = value;
    result_cancelled[ctx] = was_cancelled;
    result_valid[ctx] = 1;
    do @(posedge clock); while (!result_ready[ctx]);
    @(negedge clock);
    result_valid[ctx] = 0;
    result_task[ctx] = {$urandom, $urandom};
    result_value[ctx] = {$urandom, $urandom};
    result_cancelled[ctx] = $urandom;
  endtask
  `include "control_random.svh"
  `include "control_reset.svh"
  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    transact(0, 0, 7, 0, 2);
    management.transfer(0, 0, 0, 0, signature[0]);
    management.transfer(0, 1, 0, 0, signature[1]);
    transact(0, 0, 1, 0, 1);
    transact(0, 0, 2, 0, `CODE_BYTES);
    transact(0, 0, 3, 0, `TLS_BASE);
    transact(0, 0, 4, 0, `TLS_BYTES);
    transact(0, 0, 5, 0, `TSS_BASE);
    transact(0, 0, 6, 0, `TSS_BYTES);
    transact(5, 0, 0, 0);
    descriptor(0);
    transact(1, 0, 5, signature[0]);
    // Staging was performed with another address context. Only Start selects the task's context.
    request_satp = 64'h8000000000000123;
    request_privilege = 1;
    request_sum = 1;
    request_mxr = 0;
    start_ready[0] = 0;
    fork
      transact(2, 0, 0, 0);
      begin
        wait (start_valid[0]);
        repeat (5) begin
          @(negedge clock);
          if (!start_valid[0] || management.ready || launches != 0)
            `uvm_fatal("STALL", "launch was accepted before the thread became ready")
          if (task_id[0] !== 7 || launch_satp[0] !== request_satp)
            `uvm_fatal("STALL", "launch descriptor and address context changed under backpressure")
        end
        start_ready[0] = 1;
      end
    join
    transact(0, 0, 1, 0, 3);
    descriptor(1);
    request_satp = 64'h8000000000000456;
    request_privilege = 0;
    request_sum = 0;
    request_mxr = 1;
    transact(2, 1, 0, 0);
    @(negedge clock);
    result_valid = 3;
    do @(posedge clock); while (result_ready != 3);
    @(negedge clock);
    result_valid = 0;
    transact(0, 0, 1, 0, 5);
    transact(0, 0, 8, 0, 7);
    transact(0, 0, 9, 0, 64'h42);
    transact(0, 1, 8, 0, 8);
    transact(3, 0, 0, 0);
    transact(3, 1, 0, 0);
    transact(6, 0, 0, 0);
    if (in_use || launches != 2) `uvm_fatal("GROUP", "group lifetime or launch count wrong")
    control_random(200);
    control_reset();
    `uvm_info("CONTROL", $sformatf(
              "%0d transactions checked; %0d launches; %0d cancels", checked, launches, cancels),
              UVM_LOW)
    control.done = 1;
  end
endmodule
