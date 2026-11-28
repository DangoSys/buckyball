package scu_pkg;
  class coverage;
    covergroup behavior with function sample (int kind, bit hold);
      op: coverpoint kind {
        bins ready_rd = {0};
        bins ready_wr = {1};
        bins tx = {2};
        bins exit_wr = {3};
        bins poll_empty = {4};
        bins poll_full = {5};
        bins pop = {6};
        bins invalid = {7};
      }
      response_stall: coverpoint hold;
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage

module scu_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  import "DPI-C" function void scu_rx_input(
    input int unsigned hart,
    ch
  );
  import "DPI-C" function int unsigned scu_observed(
    input int unsigned kind,
    output int unsigned hart,
    value
  );
  logic clock = 0, reset = 1, valid = 0, ready, wr = 0, rsp_valid, rsp_ready = 0, error;
  always #5 clock = ~clock;
  logic [`SCU_TAG_BITS-1:0] tag = 0, rsp_tag;
  logic [63:0] address = 0, data = 0, result_data;
  logic [2:0] size = 0;
  int unsigned events[3], last_hart[3], last_value[3];
  int unsigned checked;
  scenario_if control (clock);
  scu_pkg::coverage cov = new();
  SystemControl dut (
      .clock(clock),
      .reset(reset),
      .io_request_valid(valid),
      .io_request_ready(ready),
      .io_request_bits_addr(address),
      .io_request_bits_tag(tag),
      .io_request_bits_size(size),
      .io_request_bits_write(wr),
      .io_request_bits_data(data),
      .io_response_valid(rsp_valid),
      .io_response_ready(rsp_ready),
      .io_response_bits_tag(rsp_tag),
      .io_response_bits_error(error),
      .io_response_bits_data(result_data)
  );
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end

  task automatic access (input int unsigned hart, offset, input bit write_enable,
                         input longint unsigned payload, input longint unsigned expected_data,
                         input int kind, input int hold_cycles);
    logic [`SCU_TAG_BITS-1:0] saved_tag;
    int unsigned h, v, n;
    valid = 1;
    wr = write_enable;
    address = `MMIO_BASE + hart * `SCU_STRIDE + offset;
    data = payload;
    size = (offset inside {'h20004, 'h20005, 'h20006}) ? 0 : 2;
    saved_tag = `SCU_TAG_BITS'(checked * 17);
    tag = saved_tag;
    do @(posedge clock); while (!ready);
    @(negedge clock);
    valid = 0;
    address = '1;
    tag = '1;
    data = '1;
    wait (rsp_valid);
    case (kind)
      2: begin
        events[0]++;
        last_hart[0]  = hart;
        last_value[0] = payload[7:0];
      end
      3: begin
        events[1]++;
        last_hart[1]  = hart;
        last_value[1] = payload[31:0];
      end
      4, 5, 6: begin
        events[2]++;
        last_hart[2]  = hart;
        last_value[2] = kind == 6;
      end
    endcase
    repeat (hold_cycles + 1) begin
      evidence::check(
          rsp_valid && !ready && rsp_tag===saved_tag &&
                      error===(kind==7) && result_data===expected_data,
          "SCU response or tag");
      foreach (events[i]) begin
        n = scu_observed(i, h, v);
        evidence::check(n == events[i] && (!n || (h == last_hart[i] && v == last_value[i])),
                        "Unexpected DPI side effect, hart or payload");
      end
      @(negedge clock);
    end
    rsp_ready = 1;
    @(posedge clock);
    @(negedge clock);
    rsp_ready = 0;
    cov.behavior.sample(kind, hold_cycles != 0);
    checked++;
  endtask

  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    access (0, 'h20006, 0, 0, 0, 0, 0);
    access (1, 'h20006, 1, 'h1a5, 0, 1, 1);
    access (0, 'h20006, 0, 0, 0, 0, 0);
    access (1, 'h20006, 0, 0, 'ha5, 0, 2);
    access (0, 'h20000, 1, 'h123400fe, 0, 2, 3);
    access (1, 0, 1, 64'h12345678_89abcdef, 0, 3, 1);
    access (1, 'h20005, 0, 0, 'h60, 4, 0);
    scu_rx_input(1, 'h7b);
    access (1, 'h20005, 0, 0, 'h61, 5, 2);
    access (1, 'h20004, 0, 0, 'h7b, 6, 0);
    access (1, 'h20005, 0, 0, 'h60, 4, 1);
    access (0, 0, 0, 0, 0, 7, 0);
    access (0, 'h20005, 1, 'hff, 0, 7, 0);
    access (0, 'h20000, 0, 0, 0, 7, 1);
    access (2, 'h20006, 0, 0, 0, 7, 1);
    reset = 1;
    repeat (2) @(negedge clock);
    reset = 0;
    access (1, 'h20006, 0, 0, 0, 0, 0);
    control.done = 1;
  end
endmodule
