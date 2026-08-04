typedef struct {
  int unsigned bid;
  int unsigned funct7;
  int unsigned iter;
  int unsigned ksize;
  int unsigned stride;
  int unsigned padding;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned num_src_words;
  int unsigned num_dst_words;
} im2col_cmd_dpi_t;

import "DPI-C" function int im2col_case_load(
  input int unsigned seed,
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void im2col_case_cmd(output im2col_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned im2col_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned im2col_case_src_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned im2col_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned im2col_case_dst_word_hi(input int unsigned word_index);

localparam int IM2COL_FUNCT7 = 7'd48;
localparam int IM2COL_MAX_WORDS = 128;
localparam int IM2COL_TIMEOUT_CYCLES = 100000;
localparam int IM2COL_SEED = 32'hCAFE_BABE;

function automatic int unsigned im2col_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
