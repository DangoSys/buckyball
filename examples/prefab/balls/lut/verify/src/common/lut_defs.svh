typedef struct {
  int unsigned iter;
  int unsigned op1_bank;
  int unsigned op2_bank;
  int unsigned wr_bank;
  int unsigned op1_group;
  int unsigned op2_group;
  int unsigned wr_group;
  int unsigned op1_col;
  int unsigned op2_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned num_src_words;
  int unsigned num_lut_words;
  int unsigned num_dst_words;
} lut_cmd_dpi_t;

import "DPI-C" function int lut_case_load(input int unsigned index);
import "DPI-C" function void lut_case_cmd(output lut_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned lut_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned lut_case_src_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned lut_case_lut_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned lut_case_lut_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned lut_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned lut_case_dst_word_hi(input int unsigned word_index);

`ifndef LUT_FUNCT7
`error "LUT_FUNCT7 must be provided by the selected Thread ballISA"
`endif

localparam int LUT_THREAD_FUNCT7 = `LUT_FUNCT7;
localparam int LUT_MAX_SRC_WORDS = 8;
localparam int LUT_SHARED_TABLE_ROWS = 16;
localparam int LUT_LANE_TABLE_ROWS = 256;
localparam int LUT_MAX_DST_WORDS = 8;
localparam int LUT_NUM_CASES = 4;
localparam int LUT_TIMEOUT_CYCLES = 20000;
