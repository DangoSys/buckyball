module `CHI_CREDIT_TB;
  import uvm_pkg::*;
  import chi_credit_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  chi_credit_if source_if (clock);
`ifdef CHI_TX_OVERFLOW
  Tx dut (
      .clock(clock),
      .reset(source_if.reset),
      .io_active(source_if.active),
      .io_in_valid(1'b0),
      .io_in_ready(),
      .io_in_bits(32'b0),
      .io_credits(),
      .io_link_flitpend(),
      .io_link_flitv(),
      .io_link_flit(),
      .io_link_lcrdv(source_if.lcrdv)
  );
`else
  Rx dut (
      .clock(clock),
      .reset(source_if.reset),
      .io_active(source_if.active),
      .io_link_flitpend(source_if.pend),
      .io_link_flitv(source_if.valid),
      .io_link_flit(source_if.flit),
      .io_link_lcrdv(source_if.lcrdv),
      .io_out_valid(),
      .io_out_ready(1'b0),
      .io_out_bits()
  );
`endif
  initial begin
    uvm_config_db#(virtual chi_credit_if)::set(null, "uvm_test_top", "vif", source_if);
    run_test("protocol_test");
  end
endmodule
