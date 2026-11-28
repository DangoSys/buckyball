package gpdomain_pkg;
  class coverage;
    covergroup behavior with function sample (int kind, bit buffer_number);
      transaction: coverpoint kind {
        bins text = {0};
        bins constants = {1};
        bins bad_size = {2};
        bins bad_header = {3};
        bins transport = {4};
      }
      image_buffer: coverpoint buffer_number;
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage

module gpdomain_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  logic start_valid = 0, start_ready, buffer_number = 0, data_valid = 0, data_ready;
  logic [31:0] bytes = 0, word_data = 0;
  logic end_valid = 0, end_ready;
  logic [ 3:0] transport_error = 0;
  logic [63:0] fault_addr = 0;
  logic done_valid, done_ready = 0, fault, release_valid = 0, release_buffer = 0;
  logic [31:0] cause;
  logic [63:0] tval;
  logic cmd_done;
  scenario_if control (clock);
  gpdomain_pkg::coverage cov = new();
  GpDomain dut (
      .clock(clock),
      .reset(reset),
      .io_ownerId(0),
      .io_command_valid(1'b0),
      .io_complete_ready(1'b0),
      .io_complete_valid(cmd_done),
      .io_layoutRequest_ready(1'b0),
      .io_layoutResponse_valid(1'b0),
      .io_image_start_valid(start_valid),
      .io_image_start_ready(start_ready),
      .io_image_start_bits_buffer(buffer_number),
      .io_image_start_bits_bytes(bytes),
      .io_image_data_valid(data_valid),
      .io_image_data_ready(data_ready),
      .io_image_data_bits(word_data),
      .io_image_terminal_valid(end_valid),
      .io_image_terminal_ready(end_ready),
      .io_image_terminal_bits_error(transport_error),
      .io_image_terminal_bits_address(fault_addr),
      .io_image_complete_valid(done_valid),
      .io_image_complete_ready(done_ready),
      .io_image_complete_bits_fault(fault),
      .io_image_complete_bits_cause(cause),
      .io_image_complete_bits_tval(tval),
      .io_image_release_valid(release_valid),
      .io_image_release_bits(release_buffer)
  );
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end
  task automatic start_image(input bit slot, input int length);
    buffer_number = slot;
    bytes = length;
    start_valid = 1;
    do @(posedge clock); while (!start_ready);
    @(negedge clock);
    start_valid = 0;
  endtask
  task automatic send_word(input int unsigned val);
    data_valid = 1;
    word_data  = val;
    do @(posedge clock); while (!data_ready);
    @(negedge clock);
    data_valid = 0;
    word_data  = '1;
  endtask
  task automatic terminal(input int unsigned err, input longint unsigned address);
    transport_error = err;
    fault_addr = address;
    end_valid = 1;
    do @(posedge clock); while (!end_ready);
    @(negedge clock);
    end_valid = 0;
  endtask
  task automatic completion(input int kind, input bit bad, input int expected_cause,
                            input longint unsigned expected_tval);
    wait (done_valid);
    repeat (3) begin
      @(negedge clock);
      evidence::check(done_valid && !start_ready && !cmd_done,
                      "Upload completed on the wrong channel or changed under stall");
      evidence::check(fault === bad && cause === expected_cause && tval === expected_tval,
                      "Image completion payload");
    end
    done_ready = 1;
    @(posedge clock);
    @(negedge clock);
    done_ready = 0;
    cov.behavior.sample(kind, buffer_number);
  endtask
  task automatic header(input int unsigned magic, input int const_bytes);
    send_word(magic);
    send_word(4);
    send_word(0);
    send_word('h40000000);
    send_word(const_bytes);
    send_word(0);
  endtask
  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    start_image(0, 28);
    header('h31564b52, 0);
    send_word('h00008067);
    repeat (4) begin
      @(negedge clock);
      evidence::check(!done_valid, "Upload completed before the transport terminal");
    end
    terminal(0, 0);
    completion(0, 0, 0, 0);
    release_buffer = 0;
    release_valid  = 1;
    @(negedge clock);
    release_valid = 0;
    start_image(1, 32);
    header('h31564b52, 4);
    send_word('h00008067);
    send_word('hffff1234);
    terminal(0, 0);
    completion(1, 0, 0, 0);
    start_image(0, 25);
    completion(2, 1, 2, 25);
    start_image(1, 28);
    header('h31564b53, 0);
    completion(3, 1, 2, 'h31564b53);
    start_image(0, 28);
    send_word('h31564b52);
    terminal(1, 'h12345678);
    completion(4, 1, 5, 'h12345678);
    start_image(1, 28);
    send_word('h31564b52);
    reset = 1;
    repeat (2) @(negedge clock);
    reset = 0;
    evidence::check(!done_valid && start_ready, "Reset did not cancel an incomplete upload");
    start_image(1, 28);
    header('h31564b52, 0);
    send_word('h00008067);
    terminal(0, 0);
    completion(0, 0, 0, 0);
    control.done = 1;
  end
endmodule
