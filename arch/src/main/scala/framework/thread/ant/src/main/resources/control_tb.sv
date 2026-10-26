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
  logic [1:0] start_valid, start_ready = '1, result_valid = 0, result_ready, cancel;
  logic storage_busy = 0, in_use;
  logic [31:0] task_id[2], result_task[2] = '{7, 8};
  logic [63:0] entry[2], code_end[2], argument[2], stack[2];
  logic [63:0] signature[2];
  logic [63:0] request_satp = 64'h8000000000000bad, launch_satp[2];
  logic [1:0] request_privilege = 0, launch_privilege[2];
  logic request_sum = 0, request_mxr = 0;
  logic [1:0] launch_sum, launch_mxr;
  logic [63:0] expected_satp[2] = '{64'h8000000000000123, 64'h8000000000000456};
  logic [1:0] expected_privilege[2] = '{2'd1, 2'd0};
  logic [1:0] expected_sum = 2'b01, expected_mxr = 2'b10;
  int checked = 0, launches = 0;
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
      .io_result_0_bits_cancelled(1'b0),
      .io_result_0_bits_value(64'h42),
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
      .io_result_1_bits_cancelled(1'b0),
      .io_result_1_bits_value(64'h42)
      `include "control_pmp.svh"
  );
  always @(posedge clock)
    if (!reset) begin
      for (int i = 0; i < 2; i++)
      if (start_valid[i] && start_ready[i]) begin
        if(task_id[i] !== 7+i || entry[i] !== 0 || code_end[i] !== 16 ||
         argument[i] !== `TLS_BASE || stack[i] !== `TLS_END)
          `uvm_fatal("START", "local descriptor changed")
        if (launch_satp[i] !== expected_satp[i] || launch_privilege[i] !== expected_privilege[i] ||
            launch_sum[i] !== expected_sum[i] || launch_mxr[i] !== expected_mxr[i])
          `uvm_fatal("CONTEXT", "launch context does not belong to the accepted start command")
        launches++;
      end
    end
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 100us);
    run_test("protocol_test");
  end
  task automatic transact(int op, int ctx, int f, longint unsigned value,
                          longint unsigned expected = 0);
    longint unsigned actual;
    management.transfer(op, ctx, f, value, actual);
    if (actual !== expected)
      `uvm_fatal("CONTROL", $sformatf("data %h expected %h", actual, expected))
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
    request_satp = expected_satp[0];
    request_privilege = expected_privilege[0];
    request_sum = expected_sum[0];
    request_mxr = expected_mxr[0];
    start_ready[0] = 0;
    fork
      transact(2, 0, 0, 0);
      begin
        wait (start_valid[0]);
        repeat (5) begin
          @(negedge clock);
          if (!start_valid[0] || management.ready || launches != 0)
            `uvm_fatal("STALL", "launch was accepted before the thread became ready")
          if (task_id[0] !== 7 || launch_satp[0] !== expected_satp[0] ||
              launch_privilege[0] !== expected_privilege[0] ||
              launch_sum[0] !== expected_sum[0] || launch_mxr[0] !== expected_mxr[0])
            `uvm_fatal("STALL", "launch descriptor and address context changed under backpressure")
        end
        start_ready[0] = 1;
      end
    join
    transact(0, 0, 1, 0, 3);
    descriptor(1);
    request_satp = expected_satp[1];
    request_privilege = expected_privilege[1];
    request_sum = expected_sum[1];
    request_mxr = expected_mxr[1];
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
    `uvm_info("CONTROL", $sformatf("%0d transactions checked; %0d launches", checked, launches),
              UVM_LOW)
    control.done = 1;
  end
endmodule
