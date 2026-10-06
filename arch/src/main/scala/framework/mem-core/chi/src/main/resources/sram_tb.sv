`ifndef CHI_SRAM_TB
`define CHI_SRAM_TB sram_tb
`endif
`ifndef CHI_SRAM_DUT
`define CHI_SRAM_DUT SramEndpoint256
`endif
module `CHI_SRAM_TB;
  import uvm_pkg::*;
  import chi_sram_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  chi_sram_if source_if (clock);
  `CHI_SRAM_DUT dut (
      .clock(clock),
      .reset(source_if.reset),
      .io_chi_rxReq_flitpend(source_if.req_pend),
      .io_chi_rxReq_flitv(source_if.req_valid),
      .io_chi_rxReq_flit(source_if.req_flit),
      .io_chi_rxReq_lcrdv(source_if.req_credit),
      .io_chi_rxDat_flitpend(source_if.dat_pend),
      .io_chi_rxDat_flitv(source_if.dat_valid),
      .io_chi_rxDat_flit(source_if.dat_flit),
      .io_chi_rxDat_lcrdv(source_if.dat_credit),
      .io_chi_txRsp_flitpend(source_if.rsp_pend),
      .io_chi_txRsp_flitv(source_if.rsp_valid),
      .io_chi_txRsp_flit(source_if.rsp_flit),
      .io_chi_txRsp_lcrdv(source_if.rsp_credit),
      .io_chi_txDat_flitpend(source_if.out_dat_pend),
      .io_chi_txDat_flitv(source_if.out_dat_valid),
      .io_chi_txDat_flit(source_if.out_dat_flit),
      .io_chi_txDat_lcrdv(source_if.out_dat_credit),
      .io_chi_rxLinkActiveReq(source_if.rx_active_req),
      .io_chi_rxLinkActiveAck(source_if.rx_active_ack),
      .io_chi_txLinkActiveReq(source_if.tx_active_req),
      .io_chi_txLinkActiveAck(source_if.tx_active_ack),
      .io_chi_rxSActive(source_if.rx_active_req),
      .io_chi_txSActive(),
      .io_outstanding(source_if.outstanding)
  );
  initial begin
    uvm_config_db#(virtual chi_sram_if)::set(null, "uvm_test_top", "vif", source_if);
    run_test("protocol_test");
  end
endmodule
