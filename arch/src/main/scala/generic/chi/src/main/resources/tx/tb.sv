module tx_tb;
  import uvm_pkg::*;
  import link_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  link_if bus (clock);
  Tx dut (
      .clock(clock),
      .reset(bus.reset),
      .io_active(bus.pins.active),
      .io_in_valid(bus.pins.input_valid),
      .io_in_ready(bus.pins.input_ready),
      .io_in_bits(bus.pins.input_data),
      .io_link_flitpend(bus.pins.pend),
      .io_link_flitv(bus.pins.output_valid),
      .io_link_flit(bus.pins.output_data),
      .io_link_lcrdv(bus.pins.credit),
      .io_credits(bus.pins.credits)
  );
  initial begin
    uvm_config_db#(virtual link_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test();
  end
endmodule
