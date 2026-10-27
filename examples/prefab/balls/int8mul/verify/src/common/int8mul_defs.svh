typedef struct {
  int unsigned iter;
  int unsigned gate_bank;
  int unsigned input_bank;
  int unsigned output_bank;
  int unsigned op1_group;
  int unsigned op2_group;
  int unsigned wr_group;
  int unsigned op1_col;
  int unsigned op2_col;
  int unsigned wr_col;
  int unsigned gate_row;
  int unsigned bank_entries;
  int unsigned rob_id;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned num_gate_words;
  int unsigned num_input_words;
  int unsigned num_dst_words;
} int8mul_cmd_dpi_t;

import "DPI-C" function int int8mul_case_load(input int unsigned index);
import "DPI-C" function void int8mul_case_cmd(output int8mul_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned int8mul_case_gate_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_gate_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_input_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_input_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_dst_word_hi(input int unsigned word_index);

`ifndef INT8MUL_FUNCT7
`error "INT8MUL_FUNCT7 must be provided by the selected Thread ballISA"
`endif

localparam int INT8MUL_THREAD_FUNCT7 = `INT8MUL_FUNCT7;
localparam int INT8MUL_MAX_ROWS = 64;
localparam int INT8MUL_NUM_CASES = 4;
localparam int INT8MUL_TIMEOUT_CYCLES = 20000;
