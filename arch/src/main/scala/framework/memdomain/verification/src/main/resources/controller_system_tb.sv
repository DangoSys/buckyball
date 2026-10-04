`include "controller_system_config.svh"
module controller_system_tb;
  import uvm_pkg::*;
  import ip_control_test_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function void admission_trace_init();
  import "DPI-C" function void dpi_bdb_set_clk(input longint unsigned cycle);
  import "DPI-C" function chandle ack_ref_create();
  import "DPI-C" function void ack_ref_destroy(input chandle model);
  import "DPI-C" function void ack_ref_write(
    input chandle model,
    input longint unsigned address,
    input longint unsigned lo,
    input longint unsigned hi,
    input int unsigned mask
  );
  import "DPI-C" function int unsigned ack_ref_check(
    input chandle model,
    input longint unsigned address,
    input longint unsigned lo,
    input longint unsigned hi
  );
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  ip_control_if ctl (clock);
  `include "controller_system_signals.svh"
ControllerSystem dut (
      `include "controller_system_ports.svh"
  );
  `include "controller_system_clocking.svh"
stream_if #(136) command_if (
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
  stream_if #(69) response_if (
      clock,
      reset
  );
  assign response_if.valid = io_core_response_valid;
  assign response_if.ready = io_core_response_ready;
  assign response_if.bits  = {io_core_response_bits_rd, io_core_response_bits_data};
  stream_if #(8) complete_if (
      clock,
      reset
  );
  assign complete_if.valid = io_core_complete_valid;
  assign complete_if.ready = io_core_complete_ready;
  assign complete_if.bits  = io_core_complete_bits_tag;
  chandle model;
  int cycle = 0, commands = 0, completions = 0, moves = 0, move_done = 0, target_writes = 0;
  bit protect = 0;
  function automatic bit [127:0] row_data(int row);
    bit [127:0] data;
    for (int j = 0; j < 16; j++) data[j*8+:8] = row * 31 + j * 7 + 3;
    return data;
  endfunction
  task automatic verify_contract(bit ok, string message);
    if (!ok) `uvm_fatal("CONTROLLER_SYSTEM", message)
  endtask
  task automatic monitor_contract();
    forever begin
      @(sample);
      if (!sample.reset) begin
        cycle++;
        dpi_bdb_set_clk(cycle);
        verify_contract(!sample.io_moveFault_valid, "real bank move reported failure");
        if (sample.io_moveAccepted) moves++;
        if (sample.io_moveCompleted) move_done++;
        if (sample.io_bankWriteAccepted_1) target_writes++;
        if (sample.io_core_command_valid && sample.io_core_command_ready) commands++;
        if (sample.io_core_complete_valid && sample.io_core_complete_ready) begin
          verify_contract(sample.io_core_complete_bits_tag == 0, "wrong completion owner");
          completions++;
        end
        if (protect)
          verify_contract(!sample.io_core_cpuAllow, "pending move/fence allowed CPU publication");
      end
    end
  endtask
  task automatic issue(int opcode, int fn, longint unsigned a, longint unsigned b,
                       longint unsigned root = 'h8000000000012345);
    @(negedge clock);
    io_core_reserve_bits_id = 0;
    io_core_reserve_valid   = 1;
    do @(sample); while (!sample.io_core_reserve_ready);
    @(negedge clock);
    io_core_reserve_valid = 0;
    io_core_command_bits_tag = 0;
    io_core_command_bits_satp = root;
    io_core_command_bits_instruction_opcode = opcode;
    io_core_command_bits_instruction_funct = fn;
    io_core_command_bits_instruction_funct3 = 3;
    io_core_command_bits_instruction_xs1 = 1;
    io_core_command_bits_instruction_xs2 = 1;
    io_core_command_bits_instruction_xd = opcode == 'h2b;
    io_core_command_bits_instruction_rd = 11;
    io_core_command_bits_instruction_rs1Data = a;
    io_core_command_bits_instruction_rs2Data = b;
    io_core_command_valid = 1;
    if (opcode == 'h7b && (fn == 13 || fn == 0)) protect = 1;
    do @(sample); while (!sample.io_core_command_ready);
    @(negedge clock);
    io_core_command_valid = 0;
  endtask
  task automatic receive_response(output longint unsigned value);
    do @(sample); while (!sample.io_core_response_valid);
    repeat (5) begin
      @(sample);
      verify_contract(sample.io_core_response_valid && sample.io_core_response_bits_rd == 11,
                      "response withdrawn/wrong rd");
    end
    value = sample.io_core_response_bits_data;
    @(negedge clock);
    io_core_response_ready = 1;
    do @(sample); while (!sample.io_core_response_valid);
    @(negedge clock);
    io_core_response_ready = 0;
  endtask
  task automatic complete_command();
    do @(sample); while (!sample.io_core_complete_valid);
    repeat (5) begin
      @(sample);
      verify_contract(sample.io_core_complete_valid && sample.io_core_complete_bits_tag == 0,
                      "completion withdrawn/wrong tag");
    end
    @(negedge clock);
    io_core_complete_ready = 1;
    do @(sample); while (!sample.io_core_complete_valid);
    @(negedge clock);
    io_core_complete_ready = 0;
    protect = 0;
  endtask
  task automatic task_call(int fn, longint unsigned a, longint unsigned b,
                           output longint unsigned value);
    issue('h2b, fn, a, b);
    @(sample);
    verify_contract(sample.io_core_cpuAllow, "task RPC unnecessarily blocked CPU memory");
    receive_response(value);
    complete_command();
  endtask
  task automatic worker_call(int fn, longint unsigned a, output longint unsigned value);
    @(negedge clock);
    io_workers_0_cmd_bits_opcode = 'h2b;
    io_workers_0_cmd_bits_funct = fn;
    io_workers_0_cmd_bits_rs1Data = a;
    io_workers_0_cmd_bits_rs2Data = 0;
    io_workers_0_cmd_bits_rd = 12;
    io_workers_0_cmd_valid = 1;
    io_workers_0_resp_ready = 0;
    do @(sample); while (!sample.io_workers_0_cmd_ready);
    @(negedge clock);
    io_workers_0_cmd_valid = 0;
    wait (io_workers_0_resp_valid);
    repeat (3) @(sample);
    value = sample.io_workers_0_resp_bits_data;
    @(negedge clock);
    io_workers_0_resp_ready = 1;
    do @(sample); while (!sample.io_workers_0_resp_valid);
    @(negedge clock);
    io_workers_0_resp_ready = 0;
  endtask
  task automatic bank0(int row, bit write, bit [127:0] data, output bit [127:0] result);
    @(negedge clock);
    io_access_0_request_bits_bank = 0;
    io_access_0_request_bits_addr = row;
    io_access_0_request_bits_write = write;
    io_access_0_request_bits_data = data;
    io_access_0_request_bits_mask = '1;
    io_access_0_request_bits_tag = 3;
    io_access_0_request_valid = 1;
    io_access_0_response_ready = 1;
    do @(sample); while (!sample.io_access_0_request_ready);
    @(negedge clock);
    io_access_0_request_valid = 0;
    do @(sample); while (!sample.io_access_0_response_valid);
    verify_contract(
        sample.io_access_0_response_bits_tag == 3 && !sample.io_access_0_response_bits_error,
        "bank0 response owner/error");
    result = sample.io_access_0_response_bits_data;
    @(negedge clock);
    io_access_0_response_ready = 0;
  endtask
  task automatic bank1(int row, output bit [127:0] result);
    @(negedge clock);
    io_access_1_request_bits_bank = 0;
    io_access_1_request_bits_addr = row;
    io_access_1_request_bits_write = 0;
    io_access_1_request_bits_mask = '1;
    io_access_1_request_bits_tag = 4;
    io_access_1_request_valid = 1;
    io_access_1_response_ready = 1;
    do @(sample); while (!sample.io_access_1_request_ready);
    @(negedge clock);
    io_access_1_request_valid = 0;
    do @(sample); while (!sample.io_access_1_response_valid);
    verify_contract(
        sample.io_access_1_response_bits_tag == 4 && !sample.io_access_1_response_bits_error,
        "bank1 response owner/error");
    result = sample.io_access_1_response_bits_data;
    @(negedge clock);
    io_access_1_response_ready = 0;
  endtask
  initial begin
    uvm_config_db#(virtual ip_control_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 1ms);
    run_test("protocol_test");
  end
  initial begin
    longint unsigned value;
    bit [127:0] actual;
    `include "controller_system_init.svh"
admission_trace_init();
    model = ack_ref_create();
    io_allowBankResponse_0 = 1;
    io_allowBankResponse_1 = 1;
    io_allowBankResponse_2 = 1;
    io_allowBankResponse_3 = 1;
    wait (ctl.start);
    repeat (5) @(negedge clock);
    reset = 0;
    io_core_cpuQuery_valid = 1;
    io_core_cpuQuery_paddr = 'h80001000;
    io_core_cpuQuery_sizeLog2 = 3;
    io_core_cpuQuery_write = 1;
    fork
      monitor_contract();
    join_none
    task_call(3, 0, 0, value);
    verify_contract(value == 4, "actual worker count");
    task_call(4, 0, 0, value);
    verify_contract(value == `CONTROLLER_SIGNATURE, "actual signature mismatch");
    for (int i = 0; i < 7; i++) begin
      task_call(7, i, i == 6 ? `CONTROLLER_SIGNATURE : 'h80000000 + i * 64, value);
      verify_contract(value == 0, "task stage failed");
    end
    task_call(0, 0, `CONTROLLER_SIGNATURE, value);
    verify_contract(value == 0, "task submit failed");
    worker_call(8, 0, value);
    verify_contract(value == 1, "worker did not receive real task");
    worker_call(9, 7, value);
    verify_contract(value == 'h8000000000012345, "TaskController did not capture controller satp");
    worker_call(9, 0, value);
    verify_contract(value == 'h80000000, "worker context corrupted");
    worker_call(2, 0, value);
    verify_contract(value == 0, "worker finish failed");
    task_call(1, 0, 0, value);
    verify_contract(value == 1, "controller did not observe success");
    for (int r = 3; r < 6; r++) begin
      bit [127:0] expected;
      expected = row_data(r);
      bank0(r, 1, expected, actual);
      ack_ref_write(model, r * 16, expected[63:0], expected[127:64], 'hffff);
    end
    fork
      begin
        wait (target_writes == 3);
        @(negedge clock);
        io_allowBankResponse_1 = 0;
        repeat (12) begin
          @(sample);
          verify_contract(
              !sample.io_core_response_valid && !sample.io_core_complete_valid && move_done == 0,
              "controller completed before final actual bank ACK");
        end
        @(negedge clock);
        io_allowBankResponse_1 = 1;
      end
      begin
        issue('h7b, 13, 64'd1 << 8, 64'd3 | (64'd7 << 16) | (64'd2 << 32));
        receive_response(value);
        verify_contract(value == 0, "MVO response failed");
        complete_command();
      end
    join
    verify_contract(moves == 1 && move_done == 1 && target_writes == 3,
                    "real MVO request/completion count");
    for (int r = 7; r < 10; r++) begin
      bank1(r, actual);
      verify_contract(ack_ref_check(model, (r - 4) * 16, actual[63:0], actual[127:64]) == 1,
                      "MVO row data mismatch");
    end
    issue('h7b, 0, 0, 0);
    complete_command();
    @(sample);
    verify_contract(sample.io_core_cpuAllow, "CPU publication did not resume after Fence");
    verify_contract(commands == completions,
                    "accepted controller command not completed exactly once");
    `uvm_info("CONTROLLER_SYSTEM", $sformatf(
              "Commands%0d completions%0d actual MVO%0d/%0d targetRows%0d; Task/satp/Bank/finalACK/Fence checked",
              commands,
              completions,
              moves,
              move_done,
              target_writes
              ), UVM_LOW)
    ack_ref_destroy(model);
    ctl.done = 1;
  end
endmodule
