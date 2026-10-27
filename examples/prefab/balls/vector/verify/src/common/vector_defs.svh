typedef struct {
  int unsigned iter;
  int unsigned op1_bank;
  int unsigned op2_bank;
  int unsigned wr_bank;
  int unsigned wr_read_bank;
  int unsigned rob_id;
  int unsigned op1_group;
  int unsigned op2_group;
  int unsigned wr_group;
} vector_cmd_dpi_t;

import "DPI-C" function int vector_case_load(input int unsigned index);
import "DPI-C" function void vector_case_cmd(output vector_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned vector_case_at_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned vector_case_at_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned vector_case_b_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned vector_case_b_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned vector_case_c_old_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned vector_case_c_old_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned vector_case_c_new_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned vector_case_c_new_word_hi(input int unsigned word_index);

`ifndef VECMAT16_FUNCT7
`error "VECMAT16_FUNCT7 must be provided by the selected ballISA"
`endif

localparam int VECMAT16_THREAD_FUNCT7 = `VECMAT16_FUNCT7;
// C is sixteen rows in four groups of four int32 columns.
localparam int VECMAT16_ROWS = 16;
localparam int VECMAT16_GROUPS = 4;
localparam int VECMAT16_NUM_CASES = 7;
localparam int VECMAT16_TIMEOUT_CYCLES = 50000;
