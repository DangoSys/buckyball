`include "parameters.svh"
interface chi_retry_if (
    input logic clock
);
  logic reset;
  logic req_in_valid, req_in_ready, req_out_valid, req_out_ready;
  logic [`CHI_REQ_BITS-1:0] req_in, req_out;
  logic rsp_in_valid, rsp_in_ready, rsp_out_valid, rsp_out_ready;
  logic [`CHI_RSP_BITS-1:0] rsp_in, rsp_out;
  logic accepted_valid;
  logic [`CHI_NODE_BITS-1:0] accepted_src;
  logic [11:0] accepted_txn;
  logic [`CHI_PENDING_BITS-1:0] pending;
  assert property (@(posedge clock) disable iff (reset)
    req_out_valid && !req_out_ready |=> req_out_valid && $stable(
      req_out
  ))
  else $fatal(1, "CHI channel changed under backpressure");
  assert property (@(posedge clock) disable iff (reset)
    rsp_out_valid && !rsp_out_ready |=> rsp_out_valid && $stable(
      rsp_out
  ))
  else $fatal(1, "CHI channel changed under backpressure");
endinterface
