package plic_pkg;
  class coverage;
    covergroup behavior with function sample (int op, bit write_access);
      register_kind: coverpoint op {
        bins priority_reg = {0};
        bins pending = {1};
        bins enable_reg = {2};
        bins threshold = {3};
        bins claim_complete = {4};
        bins invalid = {5};
      }
      direction: coverpoint write_access;
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage

module plic_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  logic clock = 0, reset = 1, valid = 0, wr = 0, error;
  always #5 clock = ~clock;
  logic [2:0] size = 2, sources = 0;
  logic [63:0] addr = 0, data = 0, value;
  logic [1:0] machine_irq, supervisor_irq;
  scenario_if control (clock);
  plic_pkg::coverage cov = new();
  Plic dut (
      .clock(clock),
      .reset(reset),
      .io_port_access_valid(valid),
      .io_port_access_bits_addr(addr),
      .io_port_access_bits_write(wr),
      .io_port_access_bits_size(size),
      .io_port_access_bits_data(data),
      .io_port_read(value),
      .io_port_error(error),
      .io_sources(sources),
      .io_meip_0(machine_irq[0]),
      .io_meip_1(machine_irq[1]),
      .io_seip_0(supervisor_irq[0]),
      .io_seip_1(supervisor_irq[1])
  );
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end

  task automatic access (input int offset, input bit write_access, input int payload,
                         input int kind, input int bytes_log2 = 2);
    addr = `MMIO_BASE + offset;
    valid = 1;
    wr = write_access;
    data = payload;
    size = bytes_log2;
    @(posedge clock);
    evidence::check(error === (kind == 5), "PLIC register decode");
    if (!write_access && kind != 5)
      evidence::check(value === 64'(unsigned'(payload)), "PLIC register value");
    cov.behavior.sample(kind, write_access);
    @(negedge clock);
    valid = 0;
  endtask

  task automatic irqs(input bit [1:0] m, input bit [1:0] s);
    evidence::check(machine_irq === m && supervisor_irq === s, "PLIC context interrupt routing");
  endtask

  initial begin
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    irqs(0, 0);
    access (0, 0, 0, 5);
    access (5, 0, 0, 5);
    access (4, 1, 7, 5, 3);
    access (4, 0, 0, 0);
    access (4, 1, 2, 0);
    access (8, 1, 2, 0);
    access (12, 1, 15, 0);
    access (12, 0, 7, 0);
    // Hart 0 M sees sources 1 and 2, hart 1 S sees only source 3.
    access ('h2000, 1, 7, 2);
    access ('h2000, 0, 6, 2);
    access ('h2180, 1, 8, 2);
    access ('h200000, 1, 2, 3);
    sources = 3'b111;
    @(negedge clock);
    irqs(0, 2);
    access ('h1000, 0, 14, 1);
    access ('h1000, 1, 0, 1);
    access ('h1000, 0, 14, 1);
    // A priority equal to the threshold does not interrupt.
    access ('h200000, 0, 2, 3);
    access ('h200000, 1, 1, 3);
    irqs(1, 2);
    access ('h200004, 0, 1, 4);
    access ('h1000, 0, 12, 1);
    access ('h200004, 0, 2, 4);
    irqs(0, 2);
    // Completion from a context which did not enable source 1 is ignored.
    access ('h203004, 1, 1, 4);
    @(negedge clock);
    access ('h1000, 0, 8, 1);
    access ('h203004, 0, 3, 4);
    irqs(0, 0);
    access ('h203004, 0, 0, 4);
    access ('h200004, 1, 1, 4);
    @(negedge clock);
    irqs(1, 0);
    sources = 0;
    access ('h200004, 0, 1, 4);
    access ('h200004, 1, 1, 4);
    access ('h200004, 1, 2, 4);
    access ('h203004, 1, 3, 4);
    @(negedge clock);
    irqs(0, 0);
    access ('h1000, 0, 0, 1);
    reset = 1;
    repeat (2) @(negedge clock);
    reset = 0;
    access ('h2000, 0, 0, 2);
    access (4, 0, 0, 0);
    access ('h200000, 0, 0, 3);
    control.done = 1;
  end
endmodule
