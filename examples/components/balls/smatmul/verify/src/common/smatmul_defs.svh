typedef struct {
  int unsigned kind;
  int unsigned rob_id;
  int unsigned op1_words;
  int unsigned op2_words;
  int unsigned rs1_lo;
  int unsigned rs1_hi;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned op1_group;
  int unsigned op2_group;
  int unsigned wr_group;
} matrix_cmd_dpi_t;

`ifndef SMATMUL_TILE_ROWS
`error "SMATMUL_TILE_ROWS must be provided from the Ball configuration"
`endif

import "DPI-C" function void smatmul_case_load(
  input int unsigned seed,
  input int unsigned index,
  input int unsigned tile_rows
);
import "DPI-C" function int unsigned smatmul_case_num_commands();
import "DPI-C" function void smatmul_case_cmd(
  input int unsigned index,
  output matrix_cmd_dpi_t cmd
);
import "DPI-C" function int unsigned smatmul_case_bias_words();
import "DPI-C" function int unsigned smatmul_case_a_words(input int unsigned block);
import "DPI-C" function int unsigned smatmul_case_b_words(input int unsigned block);
import "DPI-C" function longint unsigned smatmul_case_bias_word_lo(input int unsigned index);
import "DPI-C" function longint unsigned smatmul_case_bias_word_hi(input int unsigned index);
import "DPI-C" function longint unsigned smatmul_case_a_word_lo(input int unsigned block, input int unsigned index);
import "DPI-C" function longint unsigned smatmul_case_a_word_hi(input int unsigned block, input int unsigned index);
import "DPI-C" function longint unsigned smatmul_case_b_word_lo(input int unsigned block, input int unsigned index);
import "DPI-C" function longint unsigned smatmul_case_b_word_hi(input int unsigned block, input int unsigned index);
import "DPI-C" function int unsigned smatmul_case_num_writes();
import "DPI-C" function int unsigned smatmul_case_write_addr(input int unsigned index);
import "DPI-C" function longint unsigned smatmul_case_write_data_lo(input int unsigned index);
import "DPI-C" function longint unsigned smatmul_case_write_data_hi(input int unsigned index);

`ifndef SMATMUL_OS_FUNCT7
`error "SMATMUL_OS_FUNCT7 must be provided by the selected Thread ballISA"
`endif
`ifndef SMATMUL_BIAS_FUNCT7
`error "SMATMUL_BIAS_FUNCT7 must be provided by the selected Thread ballISA"
`endif

localparam int SMATMUL_OS_THREAD_FUNCT7 = `SMATMUL_OS_FUNCT7;
localparam int SMATMUL_BIAS_THREAD_FUNCT7 = `SMATMUL_BIAS_FUNCT7;
localparam int MATRIX_TIMEOUT_CYCLES = 200000;
localparam int MATRIX_SEED = 32'hCAFE_BABE;
