package clint_pkg;
  class coverage;
    covergroup behavior with function sample (int kind, bit wr);
      register_kind: coverpoint kind {
        bins msip = {0};
        bins cmp_lo = {1};
        bins cmp_hi = {2};
        bins cmp_full = {3};
        bins time_lo = {4};
        bins time_hi = {5};
        bins time_full = {6};
        bins invalid = {7};
      }
      direction: coverpoint wr;
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage

module clint_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  logic clock = 0, reset = 1, valid = 0, wr = 0, error;
  always #5 clock = ~clock;
  logic [63:0] address = 0, data = 0, value;
  logic [2:0] size = 0;
  logic [1:0] msip, mtip;
  scenario_if control (clock);
  clint_pkg::coverage cov = new();
  longint unsigned timer = 0, compare[2];
  bit [1:0] software;
  int ticks;

  Clint dut (
      .clock(clock),
      .reset(reset),
      .io_port_access_valid(valid),
      .io_port_access_bits_addr(address),
      .io_port_access_bits_write(wr),
      .io_port_access_bits_size(size),
      .io_port_access_bits_data(data),
      .io_port_read(value),
      .io_port_error(error),
      .io_msip_0(msip[0]),
      .io_msip_1(msip[1]),
      .io_mtip_0(mtip[0]),
      .io_mtip_1(mtip[1])
  );

  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end

  always @(posedge clock) begin
    longint unsigned off, previous, word_value, update;
    int kind, hart;
    if (reset) begin
      timer = 0;
      ticks = 0;
      software = 0;
      compare = '{default: '1};
    end else begin
      evidence::check(msip === software, "Incorrect per-hart software interrupt");
      foreach (compare[i])
      evidence::check(mtip[i] === (timer >= compare[i]), "Incorrect timer interrupt");
      previous = timer;
      if (++ticks == `TICK_CYCLES) begin
        ticks = 0;
        timer++;
      end
      if (valid) begin
        off = address - `MMIO_BASE;
        kind = 7;
        hart = 0;
        word_value = 0;
        for (int i = 0; i < 2; i++) begin
          int h;
          h = i * 2;
          if (off == 4 * h && size == 2) begin
            kind = 0;
            hart = i;
            word_value = software[i];
          end else if (off == 'h4000 + 8 * h && size inside {2, 3}) begin
            kind = size == 2 ? 1 : 3;
            hart = i;
            word_value = compare[i];
          end else if (off == 'h4004 + 8 * h && size == 2) begin
            kind = 2;
            hart = i;
            word_value = compare[i];
          end
        end
        if (off == 'hbff8 && size inside {2, 3}) begin
          kind = size == 2 ? 4 : 6;
          word_value = previous;
        end else if (off == 'hbffc && size == 2) begin
          kind = 5;
          word_value = previous;
        end
        cov.behavior.sample(kind, wr);
        evidence::check(error === (kind == 7), "Incorrect CLINT register decode");
        if (kind != 7) begin
          update = data;
          if (kind inside {1, 4}) update = {word_value[63:32], data[31:0]};
          if (kind inside {2, 5}) update = {data[31:0], word_value[31:0]};
          if (size == 2) word_value = kind inside {2, 5} ? word_value[63:32] : word_value[31:0];
          evidence::check(value === word_value, "CLINT returned a wrong register value");
          if (wr)
            case (kind)
              0: software[hart] = data[0];
              1, 2, 3: compare[hart] = update;
              4, 5, 6: timer = update;
            endcase
        end
      end
    end
  end

  task automatic access (input int off, input int bytes_log2, input bit write_enable,
                         input logic [63:0] payload);
    address = `MMIO_BASE + off;
    size = bytes_log2;
    wr = write_enable;
    data = payload;
    valid = 1;
    @(negedge clock);
    valid = 0;
  endtask

  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    access (0, 2, 1, 1);
    access (8, 2, 0, 0);
    access (8, 2, 1, 1);
    access (0, 2, 1, 0);
    access (4, 2, 0, 0);
    access (0, 3, 1, '1);
    access ('h4001, 2, 1, '1);
    access ('h4000, 1, 0, 0);
    access ('h4000, 3, 1, 8);
    access ('h4010, 3, 1, 12);
    access ('hbff8, 3, 1, 7);
    repeat (22) @(negedge clock);
    access ('h4000, 2, 0, 0);
    access ('h4004, 2, 1, 3);
    access ('h4000, 2, 1, 2);
    access ('h4000, 3, 0, 0);
    access ('h4014, 2, 0, 0);
    access ('hbffc, 2, 1, 3);
    access ('hbff8, 2, 1, 32'hffff_fffe);
    repeat (10) @(negedge clock);
    access ('hbffc, 2, 0, 0);
    access ('hbff8, 2, 0, 0);
    access ('hbff8, 3, 0, 0);
    reset = 1;
    repeat (2) @(negedge clock);
    reset = 0;
    access (8, 2, 0, 0);
    access ('h4010, 3, 0, 0);
    control.done = 1;
  end
endmodule
