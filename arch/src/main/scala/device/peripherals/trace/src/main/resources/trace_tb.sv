package trace_pkg;
  class coverage;
    covergroup behavior with function sample (int kind);
      transaction: coverpoint kind {
        bins accepted = {0};
        bins end_word = {1};
        bins wrong_direction = {2};
        bins wrong_width = {3};
        bins unaligned = {4};
        bins outside = {5};
        bins gap = {6};
        bins idle = {7};
        bins reset = {8};
      }
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage

module trace_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  import "DPI-C" function int unsigned trace_observed(
    output int unsigned channel,
    word_index,
    lo,
    hi
  );
  logic clock = 0, reset = 1, valid = 0, wr = 0, error;
  always #5 clock = ~clock;
  logic [2:0] size = 0;
  logic [63:0] address = 0, data = 0, value;
  scenario_if control (clock);
  trace_pkg::coverage cov = new();
  int unsigned expected_count;
  Trace dut (
      .clock(clock),
      .reset(reset),
      .io_port_access_valid(valid),
      .io_port_access_bits_addr(address),
      .io_port_access_bits_write(wr),
      .io_port_access_bits_size(size),
      .io_port_access_bits_data(data),
      .io_port_read(value),
      .io_port_error(error)
  );
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end

  task automatic check_access(input int offset, input bit w, input int width, input int kind,
                              input bit enable = 1, input bit in_reset = 0);
    int unsigned channel, word_index, lo, hi, observed;
    bit legal;
    reset = in_reset;
    valid = enable;
    wr = w;
    size = width;
    address = `MMIO_BASE + offset;
    data = {32'ha5c3_7e19 ^ offset, 32'hf091_ab72 + expected_count};
    legal=w && width==3 && offset>=0 && offset<`TRACE_BYTES &&
          (offset % `TRACE_STRIDE) <= 88 && (offset % 8)==0;
    @(posedge clock);
    evidence::check(error === !legal && value === 0, "Trace register response");
    @(negedge clock);
    observed = trace_observed(channel, word_index, lo, hi);
    if (legal && enable && !in_reset) begin
      expected_count++;
      evidence::check(
          channel==offset/`TRACE_STRIDE && word_index==(offset%`TRACE_STRIDE)/8 &&
                      {hi,lo}===data,
          "Trace DPI lost channel, word or payload");
    end
    evidence::check(observed == expected_count, "Trace callback count");
    cov.behavior.sample(kind);
    valid = 0;
  endtask

  initial begin
    wait (control.start);
    repeat (2) @(negedge clock);
    check_access(0, 1, 3, 8, 1, 1);
    for (int channel = 0; channel < `TRACE_BYTES / `TRACE_STRIDE; channel++)
    for (int word_index = 0; word_index < 12; word_index++)
    check_access(channel * `TRACE_STRIDE + word_index * 8, 1, 3, word_index == 11 ? 1 : 0);
    check_access(0, 0, 3, 2);
    check_access(0, 1, 2, 3);
    check_access(1, 1, 3, 4);
    check_access(-8, 1, 3, 5);
    check_access(`TRACE_BYTES, 1, 3, 5);
    check_access(96, 1, 3, 6);
    check_access(0, 1, 3, 7, 0);
    control.done = 1;
  end
endmodule
