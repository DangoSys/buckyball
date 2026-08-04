typedef struct {
  int unsigned bid;
  int unsigned funct7;
  int unsigned iter;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned elem_bits;
  int unsigned num_src_words;
  int unsigned num_dst_words;
} transpose_cmd_dpi_t;

import "DPI-C" function int transpose_case_load(
  input int unsigned seed,
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void transpose_case_cmd(output transpose_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned transpose_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned transpose_case_src_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned transpose_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned transpose_case_dst_word_hi(input int unsigned word_index);

localparam int TRANSPOSE_FUNCT7 = 7'd49;
localparam int TRANSPOSE_MAX_WORDS = 16;
localparam int TRANSPOSE_TIMEOUT_CYCLES = 2000;
localparam int TRANSPOSE_SEED = 32'hCAFE_BABE;

function automatic int unsigned transpose_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
