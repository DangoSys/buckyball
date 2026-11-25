module rx_tb;
  import uvm_pkg::*;
  import link_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  link_if bus (clock);
  Rx dut (
      .clock(clock),
      .reset(bus.reset),
      .io_active(bus.pins.active),
      .io_link_flitpend(bus.pins.pend),
      .io_link_flitv(bus.pins.input_valid),
      .io_link_flit(bus.pins.input_data),
      .io_link_lcrdv(bus.pins.credit),
      .io_out_valid(bus.pins.output_valid),
      .io_out_ready(bus.pins.output_ready),
      .io_out_bits(bus.pins.output_data)
  );
  initial begin
    uvm_config_db#(virtual link_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test();
  end
endmodule
