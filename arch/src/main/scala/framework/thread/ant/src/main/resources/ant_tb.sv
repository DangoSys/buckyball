`include "config.svh"
module ant_tb;
  import uvm_pkg::*;
  import verification_pkg::*;
  import spm_pkg::*;
  import scenario_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function longint unsigned ant_arithmetic(
    longint unsigned a,
    b,
    int unsigned operation,
    bit word_mode
  );
  logic clock = 0, reset = 1, in_use = 0;
  always #5 clock = ~clock;
  scenario_if control (clock);
  spm_if host (
      clock,
      reset
  );
  logic [1:0] start_valid = 0, start_ready, result_valid, result_ready = 0;
  logic [1:0] cancel = 0, drained = '1, was_cancelled, running;
  logic [31:0] task_id[2] = '{1, 2}, result_task[2];
  logic [63:0] value[2];
  logic [63:0] entry[2] = '{0, 0}, code_end[2] = '{16, 16};
  logic [1:0] command_valid, command_ready = '1, retired_valid;
  logic [31:0] command_instruction[2];
  logic [63:0] command_a[2], command_b[2], retired_pc[2];
  logic [1:0] transfer_valid = 0, transfer_ready, transferred_valid, transferred_ready = 0;
  logic [1:0] transfer_code = 0, transfer_to_local = 0, transferred_result;
  logic [31:0] transfer_local[2], transfer_shared[2], transfer_bytes[2];
  int checks = 0;
  AntVerification dut (
      .clock(clock),
      .reset(reset),
      .io_inUse(in_use),
      .io_rocc_0_cmd_valid(command_valid[0]),
      .io_rocc_0_cmd_ready(command_ready[0]),
      .io_rocc_0_cmd_bits_instruction_raw_inst(command_instruction[0]),
      .io_rocc_0_cmd_bits_instruction_rs1Data(command_a[0]),
      .io_rocc_0_cmd_bits_instruction_rs2Data(command_b[0]),
      .io_rocc_0_busy(!drained[0]),
      .io_rocc_0_interrupt(1'b0),
      .io_rocc_0_resp_valid(1'b0),
      .io_rocc_0_resp_bits_rd(5'd0),
      .io_rocc_0_resp_bits_data(64'd0),
      .io_retired_0_valid(retired_valid[0]),
      .io_retired_0_bits_pc(retired_pc[0]),
      .io_rocc_1_cmd_valid(command_valid[1]),
      .io_rocc_1_cmd_ready(command_ready[1]),
      .io_rocc_1_cmd_bits_instruction_raw_inst(command_instruction[1]),
      .io_rocc_1_cmd_bits_instruction_rs1Data(command_a[1]),
      .io_rocc_1_cmd_bits_instruction_rs2Data(command_b[1]),
      .io_rocc_1_busy(!drained[1]),
      .io_rocc_1_interrupt(1'b0),
      .io_rocc_1_resp_valid(1'b0),
      .io_rocc_1_resp_bits_rd(5'd0),
      .io_rocc_1_resp_bits_data(64'd0),
      .io_retired_1_valid(retired_valid[1]),
      .io_retired_1_bits_pc(retired_pc[1]),
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
      .io_running_0(running[0]),
      .io_transfer_0_valid(transfer_valid[0]),
      .io_transfer_0_ready(transfer_ready[0]),
      .io_transfer_0_bits_code(transfer_code[0]),
      .io_transfer_0_bits_toLocal(transfer_to_local[0]),
      .io_transfer_0_bits_localOffset(transfer_local[0]),
      .io_transfer_0_bits_sharedOffset(transfer_shared[0]),
      .io_transfer_0_bits_bytes(transfer_bytes[0]),
      .io_transferred_0_valid(transferred_valid[0]),
      .io_transferred_0_ready(transferred_ready[0]),
      .io_transferred_0_bits(transferred_result[0]),
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
      .io_running_1(running[1]),
      .io_transfer_1_valid(transfer_valid[1]),
      .io_transfer_1_ready(transfer_ready[1]),
      .io_transfer_1_bits_code(transfer_code[1]),
      .io_transfer_1_bits_toLocal(transfer_to_local[1]),
      .io_transfer_1_bits_localOffset(transfer_local[1]),
      .io_transfer_1_bits_sharedOffset(transfer_shared[1]),
      .io_transfer_1_bits_bytes(transfer_bytes[1]),
      .io_transferred_1_valid(transferred_valid[1]),
      .io_transferred_1_ready(transferred_ready[1]),
      .io_transferred_1_bits(transferred_result[1]),
      .io_host_request_valid(host.valid),
      .io_host_request_ready(host.ready),
      .io_host_request_bits_address(host.address),
      .io_host_request_bits_size(host.size),
      .io_host_request_bits_write(host.write),
      .io_host_request_bits_data(host.data),
      .io_host_request_bits_mask(host.mask),
      .io_host_response_valid(host.response_valid),
      .io_host_response_ready(host.response_ready),
      .io_host_response_bits_data(host.result),
      .io_host_response_bits_error(host.error)
  );
  initial begin
    #0;
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 10ms);
    run_test("protocol_test");
  end
  task automatic copy_local(int ctx, bit code, bit to_local, int offset, int bytes);
    @(negedge clock);
    transfer_code[ctx] = code;
    transfer_to_local[ctx] = to_local;
    transfer_local[ctx] = offset;
    transfer_shared[ctx] = `TSS_SCRATCH - `TSS_BASE;
    transfer_bytes[ctx] = bytes;
    transfer_valid[ctx] = 1;
    do @(posedge clock); while (!transfer_ready[ctx]);
    @(negedge clock);
    transfer_valid[ctx] = 0;
    wait (transferred_valid[ctx]);
    repeat (3) begin
      @(negedge clock);
      if (!transferred_valid[ctx] || !transferred_result[ctx])
        `uvm_fatal("TRANSFER", "transfer completion changed under backpressure")
    end
    transferred_ready[ctx] = 1;
    @(negedge clock);
    transferred_ready[ctx] = 0;
  endtask
  task automatic put(int port, longint unsigned address, logic [127:0] data, bit wide = 0);
    logic [127:0] word;
    int ctx, offset, bytes;
    if (port == 4) begin
      host.access(address, wide ? 4 : 2, 1, data, wide ? 16'hffff : 16'h000f);
    end else begin
      ctx = port / 2;
      offset = address - (port % 2 ? `TLS_BASE : 0);
      bytes = wide ? 16 : 8;
      word = data;
      if (!wide) begin
        copy_local(ctx, !(port % 2), 0, offset & ~7, 8);
        host.send(`TSS_SCRATCH, 3, 0);
        host.take(word);
        word[(offset%8)*8+:32] = data[31:0];
      end
      host.access(`TSS_SCRATCH, wide ? 4 : 3, 1, word, wide ? 16'hffff : 16'h00ff);
      copy_local(ctx, !(port % 2), 1, offset & ~7, bytes);
    end
    if (host.error) `uvm_fatal("LOAD", "TSS access failed")
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
    put(0, 0, {32'h03700513, 32'h12b5307b, 32'h00053503, 32'h00853583}, 1);
    put(0, 16, 32'h00000073);
    code_end[0] = 20;
    put(1, `TLS_BASE, {64'hfedcba9876543210, 64'h80001000}, 1);
    command_ready[0] = 0;
    in_use = 1;
    launch(0);
    wait (command_valid[0]);
    // A busy Ant must not lock the other Ant's private loader.
    put(3, `TLS_BASE, 128'd91, 1);
    repeat (7) begin
      @(negedge clock);
      if (!command_valid[0] || command_instruction[0] !== 32'h12b5307b ||
          command_a[0] !== 64'h80001000 ||
          command_b[0] !== 64'hfedcba9876543210)
        `uvm_fatal("COMMAND", "raw NPU command changed under backpressure")
    end
    drained[0] = 0;
    command_ready[0] = 1;
    @(negedge clock);
    repeat (20) begin
      @(negedge clock);
      if (result_valid[0]) `uvm_fatal("DRAIN", "task completed before NPU drained")
    end
    drained[0] = 1;
    finish(0, 55);
    in_use = 0;
    code_end[0] = 16;
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
