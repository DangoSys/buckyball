typedef struct {
  int unsigned rows;
  int unsigned a_bank;
  int unsigned b_bank;
  int unsigned c_bank;
  int unsigned a_group_base;
  int unsigned b_group;
  int unsigned c_group;
  int unsigned a_cols;
  int unsigned group;
  int unsigned first;
  int unsigned rob_id;
} f32add_cmd_dpi_t;

import "DPI-C" function int f32add_case_load(input int unsigned index);
import "DPI-C" function void f32add_case_cmd(output f32add_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned f32add_case_a_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned f32add_case_a_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned f32add_case_b_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned f32add_case_b_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned f32add_case_c_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned f32add_case_c_word_hi(input int unsigned word_index);

`ifndef F32ADD_FUNCT7
`error "F32ADD_FUNCT7 must be provided by the selected ballISA"
`endif

localparam int F32ADD_THREAD_FUNCT7 = `F32ADD_FUNCT7;
localparam int F32ADD_NUM_CASES = 5;
localparam int F32ADD_TIMEOUT_CYCLES = 50000;
