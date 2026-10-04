`include "chi_profile.svh"
// Issue H, NodeID=7, address=44, data=256, all optional buses absent.
typedef struct packed {
  bit trace_tag;
  bit [1:0] tag_op;
  bit exp_comp_ack, excl_snoop_me;
  bit [7:0] lpid;
  bit snp_attr;
  bit [3:0] mem_attr, pcrd_type;
  bit [1:0] order;
  bit allow_retry, likely_shared;
  bit [2:0] pas;
  bit [43:0] addr;
  bit [5:0] size;
  bit multi_req;
  bit [6:0] opcode;
  bit [11:0] return_txn;
  bit stash_valid;
  bit [6:0] return_nid;
  bit [11:0] txn;
  bit [6:0] src, tgt;
  bit [3:0] qos;
} req_t;
typedef struct packed {
  bit [5:0] line_id;
  bit trace_tag;
  bit [1:0] tag_op;
  bit [3:0] pcrd_type;
  bit [11:0] dbid;
  bit [2:0] busy, fwd_state, resp;
  bit [1:0] error;
  bit [4:0] opcode;
  bit [11:0] txn;
  bit [6:0] src, tgt;
  bit [3:0] qos;
} rsp_t;
typedef struct packed {
  bit [`CHI_DATA_BITS-1:0] data;
  bit [`CHI_BE_BITS-1:0] be;
  bit replicate;
  bit [1:0] num_dat;
  bit copy_at_home, trace_tag;
  bit [`CHI_TAG_UPDATE_BITS-1:0] tag_update;
  bit [`CHI_TAG_BITS-1:0] tag;
  bit [1:0] tag_op;
  bit [5:0] line_id;
  bit [1:0] data_id, ccid;
  bit [15:0] dbid;
  bit [2:0] busy;
  bit data_pull;
  bit [7:0] data_source;
  bit [2:0] resp;
  bit [1:0] error;
  bit [3:0] opcode;
  bit [6:0] home;
  bit [11:0] txn;
  bit [6:0] src, tgt;
  bit [3:0] qos;
} dat_t;
