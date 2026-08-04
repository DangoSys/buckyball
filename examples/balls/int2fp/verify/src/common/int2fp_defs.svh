typedef struct {
  int unsigned bid;
  int unsigned funct7;
  int unsigned iter;
  int unsigned scale_bits;
  int unsigned output_mode;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned num_src_words;
} int2fp_cmd_dpi_t;

import "DPI-C" function int unsigned int2fp_ref_fp32(
  input int value,
  input int unsigned scale_bits
);
import "DPI-C" function int int2fp_ref_i8(
  input int value,
  input int unsigned scale_bits
);
import "DPI-C" function int int2fp_case_load(
  input int unsigned seed,
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void int2fp_case_cmd(output int2fp_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned int2fp_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int2fp_case_src_word_hi(input int unsigned word_index);

localparam int INT2FP_FUNCT7 = 7'd52;
localparam int INT2FP_NUM_GROUPS = 4;
localparam int INT2FP_MAX_ITER = 16;
localparam int INT2FP_MAX_WORDS = INT2FP_MAX_ITER * INT2FP_NUM_GROUPS;
localparam int INT2FP_TIMEOUT_CYCLES = 4000;
localparam int INT2FP_SEED = 32'hCAFE_BABE;
localparam int INT2FP_OUTPUT_FP32 = 0;
localparam int INT2FP_OUTPUT_INT8 = 1;

function automatic int unsigned int2fp_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
