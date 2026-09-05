typedef struct {
  int unsigned relu;
  int unsigned bid;
  int unsigned iter;
  int unsigned op1_bank;
  int unsigned op2_bank;
  int unsigned wr_bank;
  int unsigned op1_col;
  int unsigned op2_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned rs1_lo;
  int unsigned rs1_hi;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned num_lhs_words;
  int unsigned num_rhs_words;
  int unsigned num_dst_words;
} int8add_cmd_dpi_t;

import "DPI-C" function int int8add_case_load(
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void int8add_case_cmd(output int8add_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned int8add_case_lhs_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8add_case_lhs_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned int8add_case_rhs_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8add_case_rhs_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned int8add_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8add_case_dst_word_hi(input int unsigned word_index);

`ifndef INT8ADD_FUNCT7
`error "INT8ADD_FUNCT7 must be provided by the selected Core ballISA"
`endif
`ifndef INT8ADD_RELU_FUNCT7
`error "INT8ADD_RELU_FUNCT7 must be provided by the selected Core ballISA"
`endif

localparam int INT8ADD_CORE_FUNCT7 = `INT8ADD_FUNCT7;
localparam int INT8ADD_RELU_CORE_FUNCT7 = `INT8ADD_RELU_FUNCT7;
localparam int INT8ADD_MAX_ROWS = 64;
localparam int INT8ADD_NUM_CASES = 2;
localparam int INT8ADD_TIMEOUT_CYCLES = 20000;

function automatic int unsigned int8add_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
