`include "consistency_config.svh"
interface consistency_control_if (
    input logic clock
);
  logic reset;
  logic cpu_allow, older_dispatch_pending, older_requests_drained;
  logic block_requester_rsp, block_requester_data;
  logic [`CONS_OUTSTANDING_BITS-1:0] outstanding;
  logic req_valid, snp_valid;
  logic [`CONS_REQ_WIDTH-1:0] req_bits;
  logic [`CONS_SNP_CHANNEL_WIDTH-1:0] snp_bits;
  bit start = 0, finished = 0;
endinterface
