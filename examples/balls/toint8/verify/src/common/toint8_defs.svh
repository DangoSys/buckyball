typedef struct {
  int unsigned kind;
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
  int unsigned input_base;
  int unsigned num_src_words;
  int unsigned num_scale_words;
  int unsigned num_dst_words;
} toint8_cmd_dpi_t;

import "DPI-C" function int toint8_case_load(
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void toint8_case_cmd(output toint8_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned toint8_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned toint8_case_src_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned toint8_case_scale_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned toint8_case_scale_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned toint8_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned toint8_case_dst_word_hi(input int unsigned word_index);
import "DPI-C" function int unsigned toint8_case_dst_addr(input int unsigned word_index);

`ifndef QUANT_F32_TO_I8_FUNCT7
`error "QUANT_F32_TO_I8_FUNCT7 must be provided by the selected Core ballISA"
`endif
`ifndef QUANT_I32_TO_I8_FUNCT7
`error "QUANT_I32_TO_I8_FUNCT7 must be provided by the selected Core ballISA"
`endif

localparam int QUANT_F32_TO_I8_CORE_FUNCT7 = `QUANT_F32_TO_I8_FUNCT7;
localparam int QUANT_I32_TO_I8_CORE_FUNCT7 = `QUANT_I32_TO_I8_FUNCT7;
localparam int TOINT8_KIND_F32 = 0;
localparam int TOINT8_KIND_I32 = 1;
localparam int TOINT8_MAX_SRC_WORDS = 16;
localparam int TOINT8_MAX_SCALE_WORDS = 4;
localparam int TOINT8_MAX_DST_WORDS = 4;
localparam int TOINT8_NUM_CASES = 8;
localparam int TOINT8_TIMEOUT_CYCLES = 4000;

function automatic int unsigned toint8_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
