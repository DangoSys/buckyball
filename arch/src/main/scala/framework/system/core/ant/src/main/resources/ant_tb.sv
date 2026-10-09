`include "config.svh"
module ant_tb;
  import uvm_pkg::*;
  import ip_pkg::*;
  import spm_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function longint unsigned ant_arithmetic(
    longint unsigned a,
    b,
    int unsigned operation,
    bit word_mode
  );
  logic clock = 0, reset = 1, in_use = 0;
  always #5 clock = ~clock;
  ip_control_if control (clock);
  virtual spm_if ports[5];
  for (genvar i = 0; i < 5; i++) begin : endpoints
    spm_if bus (
        clock,
        reset
    );
    initial ports[i] = bus;
  end
  logic [1:0] start_valid = 0, start_ready, result_valid, result_ready = 0;
  logic [1:0] cancel = 0, drained = '1, was_cancelled, running;
  logic [31:0] task_id[2] = '{1, 2}, result_task[2];
  logic [63:0] value[2];
  logic [63:0] entry[2] = '{0, 0}, code_end[2] = '{16, 16};
  logic [1:0] command_valid, command_ready = '1, response_valid = 0, response_ready;
  logic [31:0] command_instruction[2], command_task[2], response_task[2] = '{1, 2};
  logic [63:0] command_a[2], command_b[2], response_data[2] = '{64'd55, 64'd55};
  logic [4:0] response_rd[2] = '{5'd10, 5'd10};
  logic [1:0] response_error = 0, retired_valid;
  logic [63:0] retired_pc[2];
  int checks = 0;
  Execution dut (
      .clock(clock),
      .reset(reset),
      .io_inUse(in_use),
      .io_start_0_valid(start_valid[0]),
      .io_start_0_ready(start_ready[0]),
      .io_start_0_bits_task(task_id[0]),
      .io_start_0_bits_entry(entry[0]),
      .io_start_0_bits_codeEnd(code_end[0]),
      .io_start_0_bits_argument(`TLS_BASE),
      .io_start_0_bits_stack(`TLS_END),
      .io_result_0_valid(result_valid[0]),
      .io_result_0_ready(result_ready[0]),
      .io_result_0_bits_task(result_task[0]),
      .io_result_0_bits_cancelled(was_cancelled[0]),
      .io_result_0_bits_value(value[0]),
      .io_cancel_0(cancel[0]),
      .io_npuDrained_0(drained[0]),
      .io_running_0(running[0]),
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
      .io_start_1_valid(start_valid[1]),
      .io_start_1_ready(start_ready[1]),
      .io_start_1_bits_task(task_id[1]),
      .io_start_1_bits_entry(entry[1]),
      .io_start_1_bits_codeEnd(code_end[1]),
      .io_start_1_bits_argument(`TLS_BASE),
      .io_start_1_bits_stack(`TLS_END),
      .io_result_1_valid(result_valid[1]),
      .io_result_1_ready(result_ready[1]),
      .io_result_1_bits_task(result_task[1]),
      .io_result_1_bits_cancelled(was_cancelled[1]),
      .io_result_1_bits_value(value[1]),
      .io_cancel_1(cancel[1]),
      .io_npuDrained_1(drained[1]),
      .io_running_1(running[1]),
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
      .io_code_0_request_valid(endpoints[0].bus.valid),
      .io_code_0_request_ready(endpoints[0].bus.ready),
      .io_code_0_request_bits_address(endpoints[0].bus.address),
      .io_code_0_request_bits_size(endpoints[0].bus.size),
      .io_code_0_request_bits_write(endpoints[0].bus.write),
      .io_code_0_request_bits_data(endpoints[0].bus.data),
      .io_code_0_request_bits_mask(endpoints[0].bus.mask),
      .io_code_0_response_valid(endpoints[0].bus.response_valid),
      .io_code_0_response_ready(endpoints[0].bus.response_ready),
      .io_code_0_response_bits_data(endpoints[0].bus.result),
      .io_code_0_response_bits_error(endpoints[0].bus.error),
      .io_data_0_request_valid(endpoints[1].bus.valid),
      .io_data_0_request_ready(endpoints[1].bus.ready),
      .io_data_0_request_bits_address(endpoints[1].bus.address),
      .io_data_0_request_bits_size(endpoints[1].bus.size),
      .io_data_0_request_bits_write(endpoints[1].bus.write),
      .io_data_0_request_bits_data(endpoints[1].bus.data),
      .io_data_0_request_bits_mask(endpoints[1].bus.mask),
      .io_data_0_response_valid(endpoints[1].bus.response_valid),
      .io_data_0_response_ready(endpoints[1].bus.response_ready),
      .io_data_0_response_bits_data(endpoints[1].bus.result),
      .io_data_0_response_bits_error(endpoints[1].bus.error),
      .io_code_1_request_valid(endpoints[2].bus.valid),
      .io_code_1_request_ready(endpoints[2].bus.ready),
      .io_code_1_request_bits_address(endpoints[2].bus.address),
      .io_code_1_request_bits_size(endpoints[2].bus.size),
      .io_code_1_request_bits_write(endpoints[2].bus.write),
      .io_code_1_request_bits_data(endpoints[2].bus.data),
      .io_code_1_request_bits_mask(endpoints[2].bus.mask),
      .io_code_1_response_valid(endpoints[2].bus.response_valid),
      .io_code_1_response_ready(endpoints[2].bus.response_ready),
      .io_code_1_response_bits_data(endpoints[2].bus.result),
      .io_code_1_response_bits_error(endpoints[2].bus.error),
      .io_data_1_request_valid(endpoints[3].bus.valid),
      .io_data_1_request_ready(endpoints[3].bus.ready),
      .io_data_1_request_bits_address(endpoints[3].bus.address),
      .io_data_1_request_bits_size(endpoints[3].bus.size),
      .io_data_1_request_bits_write(endpoints[3].bus.write),
      .io_data_1_request_bits_data(endpoints[3].bus.data),
      .io_data_1_request_bits_mask(endpoints[3].bus.mask),
      .io_data_1_response_valid(endpoints[3].bus.response_valid),
      .io_data_1_response_ready(endpoints[3].bus.response_ready),
      .io_data_1_response_bits_data(endpoints[3].bus.result),
      .io_data_1_response_bits_error(endpoints[3].bus.error),
      .io_shared_request_valid(endpoints[4].bus.valid),
      .io_shared_request_ready(endpoints[4].bus.ready),
      .io_shared_request_bits_address(endpoints[4].bus.address),
      .io_shared_request_bits_size(endpoints[4].bus.size),
      .io_shared_request_bits_write(endpoints[4].bus.write),
      .io_shared_request_bits_data(endpoints[4].bus.data),
      .io_shared_request_bits_mask(endpoints[4].bus.mask),
      .io_shared_response_valid(endpoints[4].bus.response_valid),
      .io_shared_response_ready(endpoints[4].bus.response_ready),
      .io_shared_response_bits_data(endpoints[4].bus.result),
      .io_shared_response_bits_error(endpoints[4].bus.error)
  );
  initial begin
    #0;
    uvm_config_db#(virtual ip_control_if)::set(null, "uvm_test_top", "vif", control);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 2ms);
    run_test("protocol_test");
  end
  task automatic put(int port, longint unsigned address, logic [127:0] data, bit wide = 0);
    logic [127:0] reply;
    ports[port].send(address, wide ? 4 : 2, 1, data, wide ? 16'hffff : 16'h000f);
    ports[port].take(reply);
    if (ports[port].error) `uvm_fatal("LOAD", "management write failed")
  endtask
  task automatic launch(int id);
    @(negedge clock);
    start_valid[id] = 1;
    do @(posedge clock); while (!start_ready[id]);
    @(negedge clock);
    start_valid[id] = 0;
  endtask
  task automatic finish(int id, longint unsigned expected, bit cancelled = 0);
    wait (result_valid[id]);
    if (was_cancelled[id] !== cancelled || value[id] !== expected || result_task[id] !== task_id[id])
      `uvm_fatal("RESULT", $sformatf(
                 "context %0d task %0d: cancelled %b value %h expected %h",
                 id,
                 result_task[id],
                 was_cancelled[id],
                 value[id],
                 expected
                 ))
    repeat (3) begin
      @(negedge clock);
      if (!result_valid[id] || value[id] !== expected) `uvm_fatal("HOLD", "completion changed")
    end
    result_ready[id] = 1;
    @(negedge clock);
    result_ready[id] = 0;
    checks++;
  endtask
  initial begin : exercise
    longint unsigned operands[8] = '{
        0,
        1,
        64'hffffffffffffffff,
        64'h8000000000000000,
        64'h7fffffffffffffff,
        64'h80000000,
        64'hffffffff,
        37
    };
    logic [31:0] operation;
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    // Real RV64IM programs: LD operands, arithmetic, ECALL returning a0.
    // Golden arithmetic uses independent Rust i128/u128 and defined division corners.
    for (int w = 0; w < 2; w++)
    for (int op = 0; op < 8; op++)
    if (!w || op == 0 || op >= 4)
      for (int sample = 0; sample < 64; sample ++) begin
        operation = 32'h02b50533 | (op << 12) | (w << 3);
        put(0, 0, {32'h00000073, operation, 32'h00053503, 32'h00853583}, 1);
        put(1, `TLS_BASE, {operands[sample%8], operands[sample/8]}, 1);
        in_use = 1;
        launch(0);
        finish(0, ant_arithmetic(operands[sample/8], operands[sample%8], op, w));
        in_use = 0;
      end
    // One producer and one consumer share data then publish/observe a flag.
    put(4, `TSS_BASE, 0, 1);
    put(0, 0, {32'h0062b023, 32'h02a00313, 32'h000202b7, 32'h00000013}, 1);
    put(0, 16, {32'h02a00513, 32'h0062b423, 32'h00100313, 32'h00000013}, 1);
    put(0, 32, 32'h00000073);
    // Consumer loops while flag is zero, then returns shared data.
    put(2, 0, {32'h0002b503, 32'hfe030ee3, 32'h0082b303, 32'h000202b7}, 1);
    put(2, 16, 32'h00000073);
    code_end[0] = 36;
    code_end[1] = 20;
    in_use = 1;
    fork
      launch(0);
      launch(1);
    join
    fork
      finish(0, 42);
      finish(1, 42);
    join
    in_use = 0;
    // Identical TLS addresses hold different values in the two contexts.
    put(0, 0, {96'b0, 32'h00053503}, 1);
    put(0, 4, 32'h00000073);
    put(2, 0, {96'b0, 32'h00053503}, 1);
    put(2, 4, 32'h00000073);
    put(1, `TLS_BASE, 128'd17, 1);
    put(3, `TLS_BASE, 128'd29, 1);
    code_end[0] = 8;
    code_end[1] = 8;
    in_use = 1;
    fork
      launch(0);
      launch(1);
    join
    fork
      finish(0, 17);
      finish(1, 29);
    join
    in_use = 0;
    code_end[0] = 16;
    // Raw custom instruction and opaque DDR pointer survive submission backpressure.
    put(0, 0, {32'h00000073, 32'h12b5757b, 32'h00053503, 32'h00853583}, 1);
    put(1, `TLS_BASE, {64'hfedcba9876543210, 64'h80001000}, 1);
    command_ready[0] = 0;
    in_use = 1;
    launch(0);
    wait (command_valid[0]);
    // A busy Ant must not lock the other Ant's private loader.
    put(3, `TLS_BASE, 128'd91, 1);
    repeat (7) begin
      @(negedge clock);
      if (!command_valid[0] || command_instruction[0] !== 32'h12b5757b ||
          command_task[0] !== task_id[0] || command_a[0] !== 64'h80001000 ||
          command_b[0] !== 64'hfedcba9876543210)
        `uvm_fatal("COMMAND", "raw NPU command changed under backpressure")
    end
    drained[0] = 0;
    command_ready[0] = 1;
    @(negedge clock);
    response_valid[0] = 1;
    do @(posedge clock); while (!response_ready[0]);
    @(negedge clock);
    response_valid[0] = 0;
    repeat (20) begin
      @(negedge clock);
      if (result_valid[0]) `uvm_fatal("DRAIN", "task completed before NPU drained")
    end
    drained[0] = 1;
    finish(0, 55);
    in_use = 0;
    // Cancel an actual in-flight DIV; reusing the context must not consume its old result.
    put(0, 0, {32'h00000073, 32'h02b54533, 32'h00053503, 32'h00853583}, 1);
    put(1, `TLS_BASE, {64'd37, 64'h7fffffffffffffff}, 1);
    in_use = 1;
    launch(0);
    do @(posedge clock); while (!(retired_valid[0] && retired_pc[0] == 4));
    repeat (8) @(negedge clock);
    cancel[0] = 1;
    @(negedge clock);
    cancel[0] = 0;
    finish(0, 0, 1);
    in_use = 0;
    put(0, 0, {64'b0, 32'h00000073, 32'h00b00513}, 1);
    code_end[0] = 8;
    in_use = 1;
    launch(0);
    finish(0, 11);
    in_use = 0;
    `uvm_info("ANT", $sformatf("%0d task completions checked", checks), UVM_LOW)
    control.done = 1;
  end
endmodule
