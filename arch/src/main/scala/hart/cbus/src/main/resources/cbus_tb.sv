package cbus_pkg;
  class coverage;
    covergroup behavior with function sample (int path, bit stalled);
      route: coverpoint path {
        bins hart0 = {0}; bins hart1 = {1}; bins external = {2}; bins uncached = {3};
      }
      backpressure: coverpoint stalled;
    endgroup
    function new();
      behavior = new();
    endfunction
  endclass
endpackage

module cbus_tb;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  import scenario_pkg::*;
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  scenario_if control (clock);
  cbus_pkg::coverage cov = new();
  logic [1:0] ar_valid = 0, ar_ready, r_valid, r_ready = 0;
  logic [  3:0] r_id  [2];
  logic [255:0] r_data[2];
  logic out_valid, out_ready = 0, out_r_valid = 0, out_r_ready;
  logic [43:0] out_addr;
  logic [3:0] out_id, out_r_id = 0, slot[2];
  logic [255:0] out_data = 0;
  logic ext_valid = 0, ext_ready, ext_r_valid, ext_r_ready = 0;
  logic [  3:0] ext_r_id;
  logic [127:0] ext_r_data;
  logic
      io_valid = 0,
      io_ready,
      io_out_valid,
      io_out_ready = 0,
      io_r_valid = 0,
      io_r_ready,
      peer_io_valid,
      peer_io_ready = 0;
  logic [43:0] io_out_addr;
  logic [3:0] io_out_id, io_return_id = 0, peer_io_id;
  logic [127:0] peer_io_data;
  logic [7:0] irqs = 0, peer_irqs;
  `define HART_PEER(P, I,
                    A) \
    .P``_memory_ddr_ar_valid(ar_valid[I]), .P``_memory_ddr_ar_ready(ar_ready[I]), \
    .P``_memory_ddr_ar_bits_addr(A), .P``_memory_ddr_ar_bits_id(4'd3), \
    .P``_memory_ddr_ar_bits_len(0), .P``_memory_ddr_ar_bits_size(5), .P``_memory_ddr_ar_bits_burst(1), \
    .P``_memory_ddr_r_ready(r_ready[I]), .P``_memory_ddr_r_valid(r_valid[I]), \
    .P``_memory_ddr_r_bits_id(r_id[I]), .P``_memory_ddr_r_bits_data(r_data[I]), \
    .P``_memory_ddr_aw_valid(1'b0), .P``_memory_ddr_w_valid(1'b0), .P``_memory_ddr_b_ready(1'b0), \
    .P``_memory_uncached_aw_valid(1'b0), .P``_memory_uncached_w_valid(1'b0), .P``_memory_uncached_b_ready(1'b0)
  CBus dut (
      .clock(clock),
      .reset(reset),
      `HART_PEER(peer0, 0, 44'h80000000),
      `HART_PEER(peer1, 1, 44'h80000100),
      .peer0_memory_uncached_ar_valid(1'b0),
      .peer0_memory_uncached_r_ready(1'b0),
      .peer1_memory_uncached_ar_valid(io_valid),
      .peer1_memory_uncached_ar_ready(io_ready),
      .peer1_memory_uncached_ar_bits_addr(44'h60020000),
      .peer1_memory_uncached_ar_bits_id(4'd7),
      .peer1_memory_uncached_ar_bits_len(0),
      .peer1_memory_uncached_ar_bits_size(2),
      .peer1_memory_uncached_ar_bits_burst(1),
      .peer1_memory_uncached_r_valid(peer_io_valid),
      .peer1_memory_uncached_r_ready(peer_io_ready),
      .peer1_memory_uncached_r_bits_id(peer_io_id),
      .peer1_memory_uncached_r_bits_data(peer_io_data),
      .external_ar_valid(ext_valid),
      .external_ar_ready(ext_ready),
      .external_ar_bits_addr(44'h80000210),
      .external_ar_bits_id(4'd9),
      .external_ar_bits_len(0),
      .external_ar_bits_size(4),
      .external_ar_bits_burst(1),
      .external_r_valid(ext_r_valid),
      .external_r_ready(ext_r_ready),
      .external_r_bits_id(ext_r_id),
      .external_r_bits_data(ext_r_data),
      .external_aw_valid(1'b0),
      .external_w_valid(1'b0),
      .external_b_ready(1'b0),
      .dlink_axi_ar_valid(out_valid),
      .dlink_axi_ar_ready(out_ready),
      .dlink_axi_ar_bits_id(out_id),
      .dlink_axi_ar_bits_addr(out_addr),
      .dlink_axi_r_valid(out_r_valid),
      .dlink_axi_r_ready(out_r_ready),
      .dlink_axi_r_bits_id(out_r_id),
      .dlink_axi_r_bits_data(out_data),
      .dlink_axi_r_bits_last(1'b1),
      .dlink_axi_r_bits_resp(0),
      .dlink_axi_aw_ready(1'b0),
      .dlink_axi_w_ready(1'b0),
      .dlink_axi_b_valid(1'b0),
      .uncached_ar_valid(io_out_valid),
      .uncached_ar_ready(io_out_ready),
      .uncached_ar_bits_addr(io_out_addr),
      .uncached_ar_bits_id(io_out_id),
      .uncached_r_valid(io_r_valid),
      .uncached_r_ready(io_r_ready),
      .uncached_r_bits_id(io_return_id),
      .uncached_r_bits_data(128'h77),
      .uncached_r_bits_last(1'b1),
      .uncached_r_bits_resp(0),
      .uncached_aw_ready(1'b0),
      .uncached_w_ready(1'b0),
      .uncached_b_valid(1'b0),
      .interrupts_0_timer(irqs[0]),
      .interrupts_0_software(irqs[1]),
      .interrupts_0_external(irqs[2]),
      .interrupts_0_supervisorExternal(irqs[3]),
      .interrupts_1_timer(irqs[4]),
      .interrupts_1_software(irqs[5]),
      .interrupts_1_external(irqs[6]),
      .interrupts_1_supervisorExternal(irqs[7]),
      .peer0_interrupts_0_timer(peer_irqs[0]),
      .peer0_interrupts_0_software(peer_irqs[1]),
      .peer0_interrupts_0_external(peer_irqs[2]),
      .peer0_interrupts_0_supervisorExternal(peer_irqs[3]),
      .peer1_interrupts_0_timer(peer_irqs[4]),
      .peer1_interrupts_0_software(peer_irqs[5]),
      .peer1_interrupts_0_external(peer_irqs[6]),
      .peer1_interrupts_0_supervisorExternal(peer_irqs[7])
  );
  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "uvm_test_top", "vif", control);
    run_test("protocol_test");
  end
  initial begin
    bit [1:0] accepted, posted;
    int index;
    wait (control.start);
    repeat (3) @(negedge clock);
    reset = 0;
    for (int bitno = 0; bitno < 8; bitno++) begin
      irqs = 8'(1 << bitno);
      @(negedge clock);
      evidence::check(peer_irqs === irqs, "Interrupt lost or routed to the wrong hart");
    end
    ar_valid = 3;
    while (accepted != 3) begin
      @(posedge clock);
      accepted |= ar_valid & ar_ready;
      @(negedge clock);
      ar_valid = ~accepted;
      if (out_valid) out_ready = 1;
      if (out_ready) break;
    end
    out_ready = 1;
    while (posted != 3 || accepted != 3) begin
      @(posedge clock);
      accepted |= ar_valid & ar_ready;
      if (out_valid && out_ready) begin
        evidence::check(out_addr inside {44'h80000000, 44'h80000100},
                        "Unexpected DDR read address");
        index = (out_addr == 44'h80000100);
        evidence::check(!posted[index], "Duplicated DDR request");
        posted[index] = 1;
        slot[index]   = out_id;
      end
      @(negedge clock);
      ar_valid = ~accepted;
    end
    evidence::check(slot[0] != slot[1],
                    "Outstanding reads with identical hart IDs lost their source");
    for (int i = 1; i >= 0; i--) begin
      out_r_valid = 1;
      out_r_id = slot[i];
      out_data = 256'hfedca001 + i;
      r_ready = 2'(1 << i);
      do @(posedge clock); while (!out_r_ready);
      evidence::check(r_valid === 2'(1 << i) && r_id[i] === 4'd3 && r_data[i] === out_data,
                      "DDR response misrouted");
      @(negedge clock);
      out_r_valid = 0;
      r_ready = 0;
      cov.behavior.sample(i, 0);
    end
    out_ready = 0;
    ext_valid = 1;
    do @(posedge clock); while (!ext_ready);
    @(negedge clock);
    ext_valid = 0;
    wait (out_valid);
    out_r_id = out_id;
    repeat (3) begin
      @(negedge clock);
      evidence::check(out_valid && (out_addr & ~44'h1f) == 44'h80000200,
                      "External read changed under stall");
    end
    out_ready = 1;
    @(negedge clock);
    out_r_valid = 1;
    out_data = {128'hf00d, 128'hbad};
    do @(posedge clock); while (!out_r_ready);
    @(negedge clock);
    out_r_valid = 0;
    wait (ext_r_valid);
    evidence::check(ext_r_valid && ext_r_id === 9 && ext_r_data === 128'hf00d,
                    "External read lane or ID incorrect");
    ext_r_ready = 1;
    @(posedge clock);
    @(negedge clock);
    ext_r_ready = 0;
    cov.behavior.sample(2, 1);
    io_valid = 1;
    do @(posedge clock); while (!io_ready);
    @(negedge clock);
    io_valid = 0;
    wait (io_out_valid);
    evidence::check(io_out_addr === 44'h60020000, "Uncached request did not reach the device bus");
    io_return_id = io_out_id;
    io_out_ready = 1;
    @(negedge clock);
    io_r_valid = 1;
    peer_io_ready = 1;
    do @(posedge clock); while (!io_r_ready);
    evidence::check(peer_io_valid && peer_io_id === 7 && peer_io_data === 128'h77,
                    "Uncached response did not reach its hart");
    @(negedge clock);
    io_r_valid = 0;
    cov.behavior.sample(3, 0);
    control.done = 1;
  end
endmodule
