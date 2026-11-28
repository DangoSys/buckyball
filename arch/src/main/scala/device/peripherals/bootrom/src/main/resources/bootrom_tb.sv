package bootrom_pkg;
  import uvm_pkg::*;
  `include "uvm_macros.svh"

  class response extends uvm_sequence_item;
    bit [11:0] id;
    bit error;
    bit [511:0] data;
    `uvm_object_utils_begin(response)
      `uvm_field_int(id, UVM_DEFAULT)
      `uvm_field_int(error, UVM_DEFAULT)
      `uvm_field_int(data, UVM_DEFAULT)
    `uvm_object_utils_end
    function new(string name = "response");
      super.new(name);
    endfunction
  endclass

  typedef enum {
    image,
    padding,
    below,
    above,
    misaligned,
    write_request
  } access_kind;
  class coverage;
    covergroup behavior with function sample (access_kind kind, bit stalled);
      access: coverpoint kind;
      backpressure: coverpoint stalled;
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage

module bootrom_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  import bootrom_pkg::*;

  logic clock = 0;
  always #5 clock = ~clock;
  logic reset = 1;
  logic valid = 0, ready, write = 0, rsp_valid, rsp_ready = 0, error;
  logic [11:0] id = 0, rsp_id;
  logic [`ROM_ADDRESS_BITS-1:0] address = 0;
  logic [511:0] data;
  scenario_if control (clock);
  result_board #(response, 0) board;
  coverage cov;
  access_kind current;
  bit stalled;

  BootRomLines dut (
      .clock(clock),
      .reset(reset),
      .io_request_valid(valid),
      .io_request_ready(ready),
      .io_request_bits_id(id),
      .io_request_bits_addr(address),
      .io_request_bits_write(write),
      .io_request_bits_data('1),
      .io_request_bits_mask('1),
      .io_response_valid(rsp_valid),
      .io_response_ready(rsp_ready),
      .io_response_bits_id(rsp_id),
      .io_response_bits_data(data),
      .io_response_bits_error(error)
  );

  initial begin
    board = new("responses", null);
    cov   = new();
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end

  always @(posedge clock) begin
    if (reset) begin
      board.cancel_pending();
      stalled = 0;
    end else begin
      if (valid && ready) begin
        automatic response expected = new();
        expected.id = id;
        if (write) current = write_request;
        else if (address < `ROM_BASE) current = below;
        else if (address >= `ROM_BASE + `ROM_BYTES) current = above;
        else if (address[5:0] != 0) current = misaligned;
        else if (address - `ROM_BASE >= `ROM_IMAGE_BYTES) current = padding;
        else current = image;
        expected.error = !(current inside {image, padding});
        expected.data  = '0;
        if (!expected.error) begin
          for (int i = 0; i < 64; i++) begin
            automatic int offset = int'(address - `ROM_BASE) + i;
            if (offset < `ROM_IMAGE_BYTES) expected.data[8*i+:8] = 8'(37 * offset + 11);
          end
        end
        board.write_expected(expected);
      end
      if (rsp_valid && !rsp_ready) stalled = 1;
      if (rsp_valid && rsp_ready) begin
        automatic response observed = new();
        evidence::check(!$isunknown({rsp_id, error, data}), "Response contains unknown bits");
        observed.id = rsp_id;
        observed.error = error;
        observed.data = data;
        board.write_actual(observed);
        cov.behavior.sample(current, stalled);
        stalled = 0;
      end
    end
  end

  task automatic send(input longint unsigned addr, input bit wr, input bit [11:0] tag);
    address = addr;
    write = wr;
    id = tag;
    valid = 1;
    do @(posedge clock); while (!ready);
    @(negedge clock);
    valid = 0;
    address = '1;
    id = '1;
  endtask

  task automatic access_line(input longint unsigned addr, input bit wr, input int stalls);
    logic [524:0] held;
    send(addr, wr, 12'(board.checked * 317));
    wait (rsp_valid);
    held = {rsp_id, error, data};
    repeat (stalls) begin
      @(negedge clock);
      evidence::check(rsp_valid && {rsp_id, error, data} === held, "Held response changed");
      evidence::check(!ready, "Second request accepted while response pending");
    end
    rsp_ready = 1;
    @(posedge clock);
    @(negedge clock);
    rsp_ready = 0;
  endtask

  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    for (int offset = 0; offset < `ROM_BYTES; offset += 64)
    access_line(`ROM_BASE + offset, 0, (offset / 64) % 3);
    access_line(`ROM_BASE - 64, 0, 1);
    access_line(`ROM_BASE + `ROM_BYTES, 0, 0);
    access_line(`ROM_BASE + 1, 0, 2);
    access_line(`ROM_BASE, 1, 1);
    access_line(`ROM_BASE, 0, 0);
    send(`ROM_BASE + 64, 0, 12'habc);
    reset = 1;
    repeat (2) @(negedge clock);
    evidence::check(!rsp_valid, "Reset retained a pending response");
    reset = 0;
    access_line(`ROM_BASE + 64, 0, 1);
    evidence::check(
        board.checked == `ROM_BYTES / 64 + 6 && board.pending() == 0 && evidence::cancelled == 1,
        "Responses and reset cancellation did not reconcile");
    control.done = 1;
  end
endmodule
