module master_tb;
  import uvm_pkg::*;
  import axis_pkg::*;

  logic clock = 1'b0;
  logic reset = 1'b0;
  always #5 clock = ~clock;

  axis_if source_if (clock);
  axis_if sink_if (clock);

  assign source_if.reset = reset;
  assign sink_if.reset   = reset;

  Master dut (
      .clock(clock),
      .reset(reset),
      .io_in_ready(source_if.tready),
      .io_in_valid(source_if.tvalid),
      .io_in_bits_tdata(source_if.tdata),
      .io_in_bits_tkeep(source_if.tkeep),
      .io_in_bits_tlast(source_if.tlast),
      .io_axis_tvalid(sink_if.tvalid),
      .io_axis_tready(sink_if.tready),
      .io_axis_tdata(sink_if.tdata),
      .io_axis_tkeep(sink_if.tkeep),
      .io_axis_tlast(sink_if.tlast)
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
