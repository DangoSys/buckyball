module decoder_tb;
  import uvm_pkg::*;
  import scenario_pkg::*;
  `include "uvm_macros.svh"
  logic clock = 0, reset = 1;
  always #5 clock = ~clock;
  scenario_if ctl (clock);
  `include "decoder_signals.svh"
GlobalDecoder dut (
      `include "decoder_ports.svh"
  );
  `include "decoder_clocking.svh"
  int checks = 0;
  function automatic void check(bit good, string message);
    verification_pkg::evidence::checks++;
    checks++;
    if (!good) `uvm_fatal("DECODER", message)
  endfunction
  `include "decoder_config.svh"
  import "DPI-C" function int unsigned decoder_ref_domain(
    int unsigned funct,
    int unsigned opcode,
    longint unsigned rs1,
    int unsigned rvv,
    int unsigned banks,
    int unsigned bits
  );
  task automatic decode(int funct, int expected_bid, longint unsigned rs1, longint unsigned rs2);
    io_id_i_valid = 1;
    io_id_i_bits_cmd_funct = funct;
    io_id_i_bits_cmd_opcode = 'h7b;
    io_id_i_bits_cmd_rs1Data = rs1;
    io_id_i_bits_cmd_rs2Data = rs2;
    io_id_o_ready = 0;
    repeat (2) begin
      @(sample);
      check(sample.io_id_o_valid && !sample.io_id_i_ready, "decode backpressure mismatch");
      check(sample.io_id_o_bits_domain_id == decoder_ref_domain(
            funct, 'h7b, rs1, `DECODER_RVV, `DECODER_VBANKS, `DECODER_BANK_BITS),
            "domain decode mismatch");
      check(
          sample.io_id_o_bits_ball_bid == expected_bid && sample.io_id_o_bits_cmd_rs1Data == rs1 && sample.io_id_o_bits_cmd_rs2Data == rs2,
          "instruction payload mismatch");
      @(negedge clock);
    end
    io_id_o_ready = 1;
    @(sample);
    check(sample.io_id_i_ready, "decoder did not accept ready output");
    @(negedge clock);
    io_id_i_valid = 0;
  endtask

  initial begin
    uvm_config_db#(virtual scenario_if)::set(null, "*", "vif", ctl);
    uvm_config_db#(time)::set(null, "uvm_test_top", "timeout", 2ms);
    run_test("protocol_test");
  end
  initial begin
    `include "decoder_init.svh"
    wait (ctl.start);
    repeat (3) @(sample);
    @(negedge clock);
    reset = 0;
    decode(1, 0, 0, 0);
    decode(32, 0, 0, 1056);
    decode(33, 0, 2 << 20, 16);
    decode(34, 0, (64'h1 << 63) | (2 << 20), 16);
    decode(35, 0, 0, 16);
    decode(16, 0, 0, 16);
    decode(13, 0, 0, 0);
    `include "decoder_cases.svh"
`uvm_info("DECODER", $sformatf("%0d checked outputs", checks), UVM_LOW)
    ctl.done = 1;
  end
endmodule
