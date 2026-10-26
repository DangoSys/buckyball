`include "profile.svh"
package observed_types;
  typedef struct packed {
    logic reset;
    logic req_in_valid, req_in_ready, req_out_valid, req_out_ready;
    logic [`CHI_REQ_BITS-1:0] req_in, req_out;
    logic rsp_in_valid, rsp_in_ready, rsp_out_valid, rsp_out_ready;
    logic [`CHI_RSP_BITS-1:0] rsp_in, rsp_out;
    logic accepted_valid;
    logic [`CHI_NODE_BITS-1:0] accepted_src;
    logic [11:0] accepted_txn;
    logic [`CHI_PENDING_BITS-1:0] pending;
  } pins_t;
endpackage
