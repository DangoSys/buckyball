`include "config.svh"
module control_tb;
  import uvm_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  ip_control_if control (clock);
  ant_control_if management (
      clock,
      reset
  );
  logic [1:0] online = '1, start_valid, start_ready = '1, result_valid = 0, result_ready, cancel;
  logic storage_busy = 0, in_use;
  logic [31:0] task_id[2], result_task[2] = '{7, 8};
  logic [63:0] entry[2], code_end[2], argument[2], stack[2];
  int checked = 0, launches = 0;
  Control dut (
      .clock(clock),
      .reset(reset),
      .io_request_valid(management.valid),
      .io_request_ready(management.ready),
      .io_request_bits_operation(management.operation),
      .io_request_bits_context(management.context_id),
      .io_request_bits_field(management.field),
      .io_request_bits_data(management.data),
      .io_response_valid(management.reply_valid),
      .io_response_ready(management.reply_ready),
      .io_response_bits(management.reply),
      .io_storageBusy(storage_busy),
      .io_inUse(in_use),
      .io_signatures_0(64'h1234),
      .io_online_0(online[0]),
      .io_cancel_0(cancel[0]),
      .io_start_0_valid(start_valid[0]),
      .io_start_0_ready(start_ready[0]),
      .io_start_0_bits_task(task_id[0]),
      .io_start_0_bits_entry(entry[0]),
      .io_start_0_bits_codeEnd(code_end[0]),
      .io_start_0_bits_argument(argument[0]),
      .io_start_0_bits_stack(stack[0]),
      .io_result_0_valid(result_valid[0]),
      .io_result_0_ready(result_ready[0]),
      .io_result_0_bits_task(result_task[0]),
      .io_result_0_bits_cancelled(1'b0),
      .io_result_0_bits_value(64'h42),
      .io_signatures_1(64'h1235),
      .io_online_1(online[1]),
      .io_cancel_1(cancel[1]),
      .io_start_1_valid(start_valid[1]),
      .io_start_1_ready(start_ready[1]),
      .io_start_1_bits_task(task_id[1]),
      .io_start_1_bits_entry(entry[1]),
      .io_start_1_bits_codeEnd(code_end[1]),
      .io_start_1_bits_argument(argument[1]),
      .io_start_1_bits_stack(stack[1]),
      .io_result_1_valid(result_valid[1]),
      .io_result_1_ready(result_ready[1]),
      .io_result_1_bits_task(result_task[1]),
      .io_result_1_bits_cancelled(1'b0),
      .io_result_1_bits_value(64'h42)
  );
  always @(posedge clock)
    if (!reset) begin
      for (int i = 0; i < 2; i++)
      if (start_valid[i] && start_ready[i]) begin
        if(task_id[i] !== 7+i || entry[i] !== 0 || code_end[i] !== 16 ||
         argument[i] !== `TLS_BASE || stack[i] !== `TLS_END)
          `uvm_fatal("START", "local descriptor changed")
        launches++;
      end
    end
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "uvm_test_top", "vif", control);
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
    transact(1, ctx, 5, 64'h1234 + ctx);
  endtask
  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    transact(0, 0, 7, 0, 2);
    transact(0, 1, 0, 0, 64'h1235);
    transact(0, 0, 1, 0, 1);
    transact(0, 0, 2, 0, 256);
    transact(0, 0, 3, 0, `TLS_BASE);
    transact(0, 0, 4, 0, 256);
    transact(0, 0, 5, 0, `TSS_BASE);
    transact(0, 0, 6, 0, 256);
    transact(5, 0, 0, 0);
    descriptor(0);
    transact(1, 0, 5, 64'h1234);
    start_ready[0] = 0;
    fork
      transact(2, 0, 0, 0);
      begin
        wait (start_valid[0]);
        repeat (5) begin
          @(negedge clock);
          if (!start_valid[0] || management.ready) `uvm_fatal("STALL", "launch was not held")
        end
        start_ready[0] = 1;
      end
    join
    transact(0, 0, 1, 0, 3);
    descriptor(1);
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
