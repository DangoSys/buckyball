typedef struct {
  int unsigned iter;
  int unsigned relu;
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
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned num_src_words;
  int unsigned num_scale_words;
  int unsigned num_dst_words;
} int2fp_cmd_dpi_t;

import "DPI-C" function int int2fp_case_load(input int unsigned index);
import "DPI-C" function void int2fp_case_cmd(output int2fp_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned int2fp_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int2fp_case_src_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned int2fp_case_scale_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int2fp_case_scale_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned int2fp_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int2fp_case_dst_word_hi(input int unsigned word_index);

`ifndef INT32_TO_FP32_FUNCT7
`error "INT32_TO_FP32_FUNCT7 must be provided by the selected Thread ballISA"
`endif

localparam int INT32_TO_FP32_THREAD_FUNCT7 = `INT32_TO_FP32_FUNCT7;
localparam int INT2FP_MAX_SRC_WORDS = 48;
localparam int INT2FP_MAX_SCALE_WORDS = 12;
localparam int INT2FP_MAX_DST_WORDS = 48;
localparam int INT2FP_NUM_CASES = 8;
localparam int INT2FP_TIMEOUT_CYCLES = 20000;
