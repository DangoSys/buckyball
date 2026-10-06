module rx_tb;
  import uvm_pkg::*;
  import chi_link_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  chi_link_if source_if (clock);
  wire flitpend, flitv, lcrdv;
  wire [31:0] flit;
  Tx transmitter (
      .clock(clock),
      .reset(source_if.reset),
      .io_active(source_if.active),
      .io_in_valid(source_if.valid),
      .io_in_ready(source_if.ready),
      .io_in_bits(source_if.data),
      .io_link_flitpend(flitpend),
      .io_link_flitv(flitv),
      .io_link_flit(flit),
      .io_link_lcrdv(lcrdv),
      .io_credits(source_if.credits)
  );
  Rx dut (
      .clock(clock),
      .reset(source_if.reset),
      .io_active(source_if.active),
      .io_link_flitpend(flitpend),
      .io_link_flitv(flitv),
      .io_link_flit(flit),
      .io_link_lcrdv(lcrdv),
      .io_out_valid(source_if.out_valid),
      .io_out_ready(source_if.out_ready),
      .io_out_bits(source_if.out_data)
  );
  initial begin
    uvm_config_db#(virtual chi_link_if)::set(null, "uvm_test_top", "vif", source_if);
    run_test("protocol_test");
  end
endmodule
