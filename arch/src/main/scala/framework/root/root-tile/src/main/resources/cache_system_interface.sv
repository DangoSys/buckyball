`include "cache_system_config.svh"
interface cache_system_if (
    input logic clock
);
  logic reset, active, block_requester_rsp;
  logic access_valid[2], access_ready[2], access_write[2], access_atomicWord[2];
  logic [`COH_REQ_ADDR_WIDTH-1:0] access_addr[2];
  logic [63:0] access_data[2], result_data[2];
  logic [7:0] access_mask  [2];
  logic [3:0] access_atomic[2];
  logic result_valid[2], result_ready[2], result_error[2];
  logic [`COH_OUTSTANDING_BITS-1:0] outstanding;
  logic req_valid, rsp_valid, dat_valid, snp_valid, rx_rsp_valid, rx_dat_valid;
  logic [`COH_REQ_WIDTH-1:0] req_bits;
  logic [`COH_RSP_WIDTH-1:0] rsp_bits, rx_rsp_bits;
  logic [`COH_DAT_WIDTH-1:0] dat_bits, rx_dat_bits;
  logic [`COH_SNP_CHANNEL_WIDTH-1:0] snp_bits;
  clocking sample @(posedge clock);
    default input #1step;
    input reset,active,block_requester_rsp,access_atomicWord,access_valid,access_ready,access_write,access_addr,access_data,access_mask,access_atomic;
    input result_valid, result_ready, result_error, result_data, outstanding;
    input req_valid, rsp_valid, dat_valid, snp_valid, rx_rsp_valid, rx_dat_valid;
    input req_bits, rsp_bits, dat_bits, snp_bits, rx_rsp_bits, rx_dat_bits;
  endclocking
  for (genvar core = 0; core < 2; core++) begin
    assert property (@(posedge clock) disable iff(reset) access_valid[core]&&!access_ready[core] |=> access_valid[core]&&$stable(
        {access_addr[core],access_write[core],access_data[core],access_mask[core],access_atomic[core],access_atomicWord[core]}
    ))
    else $fatal(1, "CacheSystem client changed stalled command");
    assert property (@(posedge clock) disable iff(reset) result_valid[core]&&!result_ready[core] |=> result_valid[core]&&$stable(
        {result_data[core], result_error[core]}
    ))
    else $fatal(1, "CacheSystem changed stalled result");
  end
endinterface
