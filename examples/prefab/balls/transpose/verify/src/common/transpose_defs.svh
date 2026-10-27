typedef struct {
  int unsigned iter;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_group;
  int unsigned wr_group;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned elem_bits;
  int unsigned num_src_words;
  int unsigned num_dst_words;
} transpose_cmd_dpi_t;

import "DPI-C" function int transpose_case_load(
  input int unsigned seed,
  input int unsigned index
);
import "DPI-C" function void transpose_case_cmd(output transpose_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned transpose_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned transpose_case_src_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned transpose_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned transpose_case_dst_word_hi(input int unsigned word_index);

`ifndef TRANSPOSE_FUNCT7
`error "TRANSPOSE_FUNCT7 must be provided by the selected Thread ballISA"
`endif
localparam int TRANSPOSE_THREAD_FUNCT7 = `TRANSPOSE_FUNCT7;
localparam int TRANSPOSE_MAX_WORDS = 32;
localparam int TRANSPOSE_TIMEOUT_CYCLES = 8000;
localparam int TRANSPOSE_SEED = 32'hCAFE_BABE;
