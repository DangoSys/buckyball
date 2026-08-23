typedef struct {
  int unsigned bid;
  int unsigned iter;
  int unsigned da_bits;
  int unsigned dw_addr;
  int unsigned dw_bits;
  int unsigned per_channel;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned num_src_words;
} int2fp_cmd_dpi_t;

import "DPI-C" function int unsigned int2fp_ref_fp32(
  input int value,
  input int unsigned da_bits,
  input int unsigned dw_bits
);
import "DPI-C" function int int2fp_case_load(
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void int2fp_case_cmd(output int2fp_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned int2fp_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int2fp_case_src_word_hi(input int unsigned word_index);
import "DPI-C" function int unsigned int2fp_case_dw_bits(input int unsigned index);

`ifndef INT2FP_TENSOR_FUNCT7
`error "INT2FP_TENSOR_FUNCT7 must be provided by the selected Core ballISA"
`endif
`ifndef INT2FP_CHANNEL_FUNCT7
`error "INT2FP_CHANNEL_FUNCT7 must be provided by the selected Core ballISA"
`endif
localparam int INT2FP_TENSOR_CORE_FUNCT7 = `INT2FP_TENSOR_FUNCT7;
localparam int INT2FP_CHANNEL_CORE_FUNCT7 = `INT2FP_CHANNEL_FUNCT7;
localparam int INT2FP_NUM_GROUPS = 4;
localparam int INT2FP_MAX_ITER = 16;
localparam int INT2FP_MAX_WORDS = INT2FP_MAX_ITER * INT2FP_NUM_GROUPS;
localparam int INT2FP_TIMEOUT_CYCLES = 4000;

function automatic int unsigned int2fp_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
