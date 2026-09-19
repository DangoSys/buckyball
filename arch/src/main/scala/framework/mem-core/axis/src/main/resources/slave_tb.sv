module slave_tb;
  import uvm_pkg::*;
  import axis_pkg::*;

  logic clock = 1'b0;
  logic reset = 1'b0;
  always #5 clock = ~clock;

  axis_if source_if (clock);
  axis_if sink_if (clock);

  assign source_if.reset = reset;
  assign sink_if.reset   = reset;

  Slave dut (
      .clock(clock),
      .reset(reset),
      .io_axis_tvalid(source_if.tvalid),
      .io_axis_tready(source_if.tready),
      .io_axis_tdata(source_if.tdata),
      .io_axis_tkeep(source_if.tkeep),
      .io_axis_tlast(source_if.tlast),
      .io_out_ready(sink_if.tready),
      .io_out_valid(sink_if.tvalid),
      .io_out_bits_tdata(sink_if.tdata),
      .io_out_bits_tkeep(sink_if.tkeep),
      .io_out_bits_tlast(sink_if.tlast)
  );

  initial begin
    reset = 1'b1;
    repeat (4) @(posedge clock);
    reset = 1'b0;
  end

  initial begin
    uvm_config_db#(virtual axis_if)::set(null, "uvm_test_top.env", "source_vif", source_if);
    uvm_config_db#(virtual axis_if)::set(null, "uvm_test_top.env", "sink_vif", sink_if);
    run_test("protocol_test");
  end
endmodule
