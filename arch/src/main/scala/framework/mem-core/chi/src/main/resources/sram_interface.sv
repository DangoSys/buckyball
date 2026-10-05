`include "chi_profile.svh"
interface chi_sram_if (
    input logic clock
);
  logic reset;
  logic req_pend, req_valid, req_credit;
  logic [136:0] req_flit;
  logic dat_pend, dat_valid, dat_credit;
  logic [`CHI_DAT_BITS-1:0] dat_flit;
  logic rsp_pend, rsp_valid, rsp_credit;
  logic [70:0] rsp_flit;
  logic out_dat_pend, out_dat_valid, out_dat_credit;
  logic [`CHI_DAT_BITS-1:0] out_dat_flit;
  logic rx_active_req, rx_active_ack, tx_active_req, tx_active_ack;
  logic [3:0] outstanding;
endinterface
