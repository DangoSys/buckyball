typedef struct {
  int unsigned bid;
  int unsigned iter;
  int unsigned da_bits;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned num_src_words;
} toint8_cmd_dpi_t;

import "DPI-C" function int toint8_ref_i8(
  input int unsigned fp_bits,
  input int unsigned scale_bits
);
import "DPI-C" function int unsigned toint8_quant_scale_bits(
  input int unsigned da_bits
);
import "DPI-C" function int toint8_case_load(
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void toint8_case_cmd(output toint8_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned toint8_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned toint8_case_src_word_hi(input int unsigned word_index);

`ifndef FP2INT_FUNCT7
`error "FP2INT_FUNCT7 must be provided by the selected Core ballISA"
`endif
localparam int FP2INT_CORE_FUNCT7 = `FP2INT_FUNCT7;
localparam int FP2INT_NUM_WORDS = 4;
localparam int FP2INT_MAX_SOURCE_WORDS = 16;
localparam int FP2INT_MAX_OUTPUT_WORDS = FP2INT_MAX_SOURCE_WORDS / FP2INT_NUM_WORDS;
localparam int FP2INT_TIMEOUT_CYCLES = 4000;

function automatic int unsigned toint8_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
