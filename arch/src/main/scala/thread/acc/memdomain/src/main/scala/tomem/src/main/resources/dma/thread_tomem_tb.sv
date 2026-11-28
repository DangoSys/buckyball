`include "parameters.svh"
package thread_tomem_pkg;
  class coverage;
    covergroup behavior with function sample (int kind);
      operation: coverpoint kind {
        bins load_rows = {0}; bins rejected_read = {1}; bins store = {2}; bins invalid_shared = {3};
      }
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage
module thread_tomem_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  logic [ `TM_INPUT_BITS-1:0] inputs = '0;
  wire  [`TM_OUTPUT_BITS-1:0] outputs;
  scenario_if control (clock);
  thread_tomem_pkg::coverage cov = new();
  ToMem dut (
      .clock(clock),
      .reset(reset),
      `include "ports.svh"
  );
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end
  task automatic request(input int tag, input bit store, input int rows, input bit shared_bank = 0);
    `TM_IO_REQUEST_BITS_CMD_IS_STORE = store;
    `TM_IO_REQUEST_BITS_CMD_IS_SHARED = shared_bank;
    `TM_IO_REQUEST_BITS_CMD_ITER = rows;
    `TM_IO_REQUEST_BITS_CMD_BANK_ID = 3;
    `TM_IO_REQUEST_BITS_CMD_MEM_ADDR = 'h80200400;
    `TM_IO_REQUEST_BITS_CMD_SPECIAL = 64'h1 << 39;
    `TM_IO_REQUEST_BITS_ROB_ID = tag;
    `TM_IO_REQUEST_VALID = 1;
    do @(posedge clock); while (!`TM_IO_REQUEST_READY);
    @(negedge clock);
    `TM_IO_REQUEST_VALID = 0;
  endtask
  task automatic read_offer(input int rows);
    wait (`TM_IO_DMA_READ_VALID);
    repeat (2) begin
      @(negedge clock);
      evidence::check(
          `TM_IO_DMA_READ_VALID && `TM_IO_DMA_READ_BITS_TRANSFER_VADDR==='h80200400 &&
          `TM_IO_DMA_READ_BITS_TRANSFER_LEN===rows*16 && `TM_IO_DMA_READ_BITS_TRANSFER_STRIDE===1,
          "Incorrect logical read offer");
      evidence::check(`TM_IO_MEMORY_CONTROL_READ_VALID && !`TM_IO_QUERYSHARED,
                      "Incorrect memory-control publication");
    end
    `TM_IO_DMA_READ_READY = 1;
    @(posedge clock);
    @(negedge clock);
    `TM_IO_DMA_READ_READY = 0;
  endtask
  task automatic complete(input int tag, input int code, input longint unsigned addr);
    wait (`TM_IO_RESPONSE_VALID);
    repeat (2) begin
      @(negedge clock);
      evidence::check(
          `TM_IO_RESPONSE_VALID && `TM_IO_RESPONSE_BITS_ROB_ID===tag &&
          `TM_IO_RESPONSE_BITS_FAULT_ERROR===code && `TM_IO_RESPONSE_BITS_FAULT_ADDRESS===addr,
          "Incorrect or unstable terminal result");
    end
    `TM_IO_RESPONSE_READY = 1;
    @(posedge clock);
    @(negedge clock);
    `TM_IO_RESPONSE_READY = 0;
  endtask
  task automatic load_beat(input int row, input logic [127:0] data, input bit last);
    `TM_IO_DMA_READRESULT_VALID = 1;
    `TM_IO_DMA_READRESULT_BITS_DATA = data;
    `TM_IO_DMA_READRESULT_BITS_LAST = last;
    `TM_IO_DMA_READRESULT_BITS_ADDRCOUNTER = row;
    @(negedge clock);
    evidence::check(!`TM_IO_DMA_READRESULT_READY && `TM_IO_BANKWRITE_WRITE_REQ_VALID,
                    "Bank backpressure was lost");
    `TM_IO_BANKWRITE_WRITE_REQ_READY = 1;
    @(posedge clock);
    evidence::check(
        `TM_IO_DMA_READRESULT_READY && `TM_IO_BANKWRITE_WRITE_REQ_BITS_ADDR===row &&
        `TM_IO_BANKWRITE_WRITE_REQ_BITS_DATA===data && `TM_IO_BANKWRITE_BANK_ID===3 &&
        `TM_IO_BANKWRITE_GROUP_ID===0 && `TM_IO_BANKWRITE_ROB_ID===4,
        "Wrong bank write");
    @(negedge clock);
    `TM_IO_DMA_READRESULT_VALID = 0;
    `TM_IO_BANKWRITE_WRITE_REQ_READY = 0;
    `TM_IO_BANKWRITE_WRITE_RESP_VALID = 1;
    `TM_IO_BANKWRITE_WRITE_RESP_BITS_OK = 1;
    @(posedge clock);
    evidence::check(`TM_IO_BANKWRITE_WRITE_RESP_READY, "Bank response was lost");
    @(negedge clock);
    `TM_IO_BANKWRITE_WRITE_RESP_VALID = 0;
  endtask
  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    `TM_IO_QUERYGROUPS = 1;
    `TM_IO_OWNERID = 7;
    request(4, 0, 2);
    read_offer(2);
    load_beat(0, 128'habcd0123456789, 0);
    load_beat(1, 128'hcdefedcba43210, 1);
    wait (`TM_IO_MEMORY_CONTROL_FINISHED_0_VALID);
    evidence::check(!`TM_IO_RESPONSE_VALID && `TM_IO_MEMORY_CONTROL_FINISHED_0_BITS_ERROR === 0,
                    "Read acknowledged before hart completion");
    `TM_IO_MEMORY_CONTROL_COMPLETE_0_VALID = 1;
    complete(4, 0, 0);
    `TM_IO_MEMORY_CONTROL_COMPLETE_0_VALID = 0;
    cov.behavior.sample(0);
    request(5, 0, 1);
    read_offer(1);
    `TM_IO_DMA_READRESULT_VALID = 1;
    `TM_IO_DMA_READRESULT_BITS_LAST = 1;
    `TM_IO_DMA_READRESULT_BITS_FAULT_ERROR = 2;
    `TM_IO_DMA_READRESULT_BITS_FAULT_ADDRESS = 'h80200400;
    @(posedge clock);
    evidence::check(`TM_IO_DMA_READRESULT_READY && !`TM_IO_BANKWRITE_WRITE_REQ_VALID,
                    "Faulting read modified bank");
    @(negedge clock);
    `TM_IO_DMA_READRESULT_VALID = 0;
    wait (`TM_IO_MEMORY_CONTROL_FINISHED_0_VALID);
    evidence::check(`TM_IO_MEMORY_CONTROL_FINISHED_0_BITS_ERROR === 2, "Read error not published");
    `TM_IO_MEMORY_CONTROL_COMPLETE_0_VALID = 1;
    `TM_IO_MEMORY_CONTROL_COMPLETE_0_BITS_ERROR = 2;
    `TM_IO_MEMORY_CONTROL_COMPLETE_0_BITS_ADDRESS = 'h80200400;
    complete(5, 2, 'h80200400);
    `TM_IO_MEMORY_CONTROL_COMPLETE_0_VALID = 0;
    cov.behavior.sample(1);
    request(6, 1, 1);
    wait (`TM_IO_DMA_WRITE_VALID);
    evidence::check(`TM_IO_DMA_WRITE_BITS_VADDR === 'h80200400 && `TM_IO_DMA_WRITE_BITS_LEN === 16,
                    "Write address or size");
    `TM_IO_DMA_WRITE_READY = 1;
    @(posedge clock);
    @(negedge clock);
    `TM_IO_DMA_WRITE_READY = 0;
    wait (`TM_IO_BANKREAD_READ_REQ_VALID);
    `TM_IO_BANKREAD_READ_REQ_READY = 1;
    @(posedge clock);
    evidence::check(`TM_IO_BANKREAD_BANK_ID === 3 && `TM_IO_BANKREAD_READ_REQ_BITS_ADDR === 0,
                    "Wrong bank read address");
    @(negedge clock);
    `TM_IO_BANKREAD_READ_REQ_READY = 0;
    `TM_IO_BANKREAD_READ_RESP_VALID = 1;
    `TM_IO_BANKREAD_READ_RESP_BITS_DATA = 128'hf123456789abcdef;
    wait (`TM_IO_DMA_WRITEDATA_VALID);
    @(negedge clock);
    evidence::check(
        !`TM_IO_BANKREAD_READ_RESP_READY && `TM_IO_DMA_WRITEDATA_BITS_LAST &&
                    `TM_IO_DMA_WRITEDATA_BITS_DATA===128'hf123456789abcdef,
        "Store data or backpressure");
    `TM_IO_DMA_WRITEDATA_READY = 1;
    @(posedge clock);
    @(negedge clock);
    `TM_IO_DMA_WRITEDATA_READY = 0;
    `TM_IO_BANKREAD_READ_RESP_VALID = 0;
    `TM_IO_DMA_WRITERESULT_VALID = 1;
    `TM_IO_DMA_WRITERESULT_BITS_DONE = 1;
    do @(posedge clock); while (!`TM_IO_DMA_WRITERESULT_READY);
    @(negedge clock);
    `TM_IO_DMA_WRITERESULT_VALID = 0;
    wait (`TM_IO_MEMORY_CONTROL_FINISHED_1_VALID);
    `TM_IO_MEMORY_CONTROL_COMPLETE_1_VALID = 1;
    complete(6, 0, 0);
    `TM_IO_MEMORY_CONTROL_COMPLETE_1_VALID = 0;
    cov.behavior.sample(2);
    request(7, 0, 1, 1);
    wait (`TM_IO_RESPONSE_VALID);
    evidence::check(!`TM_IO_DMA_READ_VALID, "Shared bank was exposed to normal DMA");
    complete(7, 5, 'h80200400);
    cov.behavior.sample(3);
    control.done = 1;
  end
endmodule
