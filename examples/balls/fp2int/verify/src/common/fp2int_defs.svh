typedef struct {
  int unsigned bid;
  int unsigned funct7;
  int unsigned iter;
  int unsigned scale_bits;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned num_src_words;
} fp2int_cmd_dpi_t;

import "DPI-C" function int fp2int_ref_i32(
  input int unsigned fp_bits,
  input int unsigned scale_bits
);
import "DPI-C" function int fp2int_ref_i8(
  input int unsigned fp_bits,
  input int unsigned scale_bits
);
import "DPI-C" function int fp2int_case_load(
  input int unsigned seed,
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void fp2int_case_cmd(output fp2int_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned fp2int_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned fp2int_case_src_word_hi(input int unsigned word_index);

localparam int FP2INT_FUNCT7 = 7'd51;
localparam int FP2INT_NUM_WORDS = 4;
localparam int FP2INT_NUM_GROUPS = 4;
localparam int FP2INT_MAX_WORDS = FP2INT_NUM_WORDS * FP2INT_NUM_GROUPS;
localparam int FP2INT_TIMEOUT_CYCLES = 400;
localparam int FP2INT_SEED = 32'hBEEF_0001;

function automatic int unsigned fp2int_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
