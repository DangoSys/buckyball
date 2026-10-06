`include "cache_system_config.svh"
interface tile_if (
    input logic clock
);
  logic reset;
  logic retired[2], trapped[2];
  logic [63:0] retiredPc[2], trapCause[2], trapValue[2], trapPc[2];
  logic [2:0] outstanding;
  logic req_valid, rsp_valid, dat_valid, snp_valid, rx_rsp_valid, rx_dat_valid;
  logic [`COH_REQ_WIDTH-1:0] req_bits;
  logic [`COH_RSP_WIDTH-1:0] rsp_bits, rx_rsp_bits;
  logic [`COH_DAT_WIDTH-1:0] dat_bits, rx_dat_bits;
  logic [`COH_SNP_CHANNEL_WIDTH-1:0] snp_bits;
  clocking sample @(posedge clock);
    default input #1step;
    input reset, retired, retiredPc, trapped, trapCause, trapValue, trapPc, outstanding;
    input req_valid, rsp_valid, dat_valid, snp_valid, rx_rsp_valid, rx_dat_valid;
    input req_bits, rsp_bits, dat_bits, snp_bits, rx_rsp_bits, rx_dat_bits;
  endclocking
endinterface
