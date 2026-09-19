module arbiter_tb;
  import uvm_pkg::*;
  import arbiter_pkg::*;

  logic clock = 1'b0;
  logic reset = 1'b0;
  always #5 clock = ~clock;

  arbiter_if source_if(clock);
  assign source_if.reset = reset;

  PacketArbiter dut (
    .clock(clock),
    .reset(reset),
    .io_in_0_ready(source_if.in_ready[0]),
    .io_in_0_valid(source_if.in_valid[0]),
    .io_in_0_bits_tdata(source_if.in_data[0]),
    .io_in_0_bits_tkeep(source_if.in_keep[0]),
    .io_in_0_bits_tlast(source_if.in_last[0]),
    .io_in_1_ready(source_if.in_ready[1]),
    .io_in_1_valid(source_if.in_valid[1]),
    .io_in_1_bits_tdata(source_if.in_data[1]),
    .io_in_1_bits_tkeep(source_if.in_keep[1]),
    .io_in_1_bits_tlast(source_if.in_last[1]),
    .io_out_ready(source_if.out_ready),
    .io_out_valid(source_if.out_valid),
    .io_out_bits_tdata(source_if.out_data),
    .io_out_bits_tkeep(source_if.out_keep),
    .io_out_bits_tlast(source_if.out_last)
  );

  initial begin
    reset = 1'b1;
    repeat (4) @(posedge clock);
    reset = 1'b0;
  end

  initial begin
    uvm_config_db#(virtual arbiter_if)::set(null, "*", "vif", source_if);
    run_test("protocol_test");
  end
endmodule
