module bankset_tb;
  import uvm_pkg::*;
  import bankset_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  port_if control (
      clock,
      timing.reset
  );
  axis_if source_if (clock);
  axis_if sink_if (clock);
  assign source_if.reset = control.reset;
  assign sink_if.reset   = control.reset;
  BankSet dut (
      .clock(clock),
      .reset(control.reset),
      .io_command_valid(control.valid),
      .io_command_ready(control.ready),
      .io_command_bits_write(control.write),
      .io_command_bits_addr(control.addr),
      .io_command_bits_beats(control.beats),
      .io_write_valid(source_if.tvalid),
      .io_write_ready(source_if.tready),
      .io_write_bits_tdata(source_if.tdata),
      .io_write_bits_tkeep(source_if.tkeep),
      .io_write_bits_tlast(source_if.tlast),
      .io_read_valid(sink_if.tvalid),
      .io_read_ready(sink_if.tready),
      .io_read_bits_tdata(sink_if.tdata),
      .io_read_bits_tkeep(sink_if.tkeep),
      .io_read_bits_tlast(sink_if.tlast),
      .io_done_valid(control.done_valid),
      .io_done_ready(control.done_ready),
      .io_done_bits(control.error)
  );
  initial begin
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top.env.reset", "timing", timing);
    uvm_config_db#(virtual port_if)::set(null, "uvm_test_top*", "control", control);
    uvm_config_db#(virtual axis_if)::set(null, "uvm_test_top.env.commands.monitor", "write_vif",
                                         source_if);
    uvm_config_db#(virtual axis_if)::set(null, "uvm_test_top.env.source.*", "vif", source_if);
    uvm_config_db#(virtual axis_if)::set(null, "uvm_test_top.env.sink", "vif", sink_if);
    uvm_config_db#(virtual axis_if)::set(null, "uvm_test_top.env.output_monitor", "vif", sink_if);
    uvm_config_db#(virtual axis_if)::set(null, "uvm_test_top.env.commands.monitor", "read_vif",
                                         sink_if);
    run_test("protocol_test");
  end
endmodule
