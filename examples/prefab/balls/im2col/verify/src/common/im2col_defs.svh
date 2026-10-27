typedef struct {
  int unsigned iter;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_group;
  int unsigned wr_group;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned num_src;
  int unsigned num_reads;
  int unsigned num_dst;
} im2col_cmd_dpi_t;

import "DPI-C" function int im2col_case_load(input int unsigned index);
import "DPI-C" function void im2col_case_cmd(output im2col_cmd_dpi_t cmd);
import "DPI-C" function int unsigned im2col_case_src_group(input int unsigned i);
import "DPI-C" function int unsigned im2col_case_src_addr(input int unsigned i);
import "DPI-C" function longint unsigned im2col_case_src_lo(input int unsigned i);
import "DPI-C" function longint unsigned im2col_case_src_hi(input int unsigned i);
import "DPI-C" function int unsigned im2col_case_read_group(input int unsigned i);
import "DPI-C" function int unsigned im2col_case_read_addr(input int unsigned i);
import "DPI-C" function int unsigned im2col_case_dst_group(input int unsigned i);
import "DPI-C" function int unsigned im2col_case_dst_addr(input int unsigned i);
import "DPI-C" function longint unsigned im2col_case_dst_lo(input int unsigned i);
import "DPI-C" function longint unsigned im2col_case_dst_hi(input int unsigned i);

`ifndef IM2COL_FUNCT7
`error "IM2COL_FUNCT7 must be provided by the selected Thread ballISA"
`endif
localparam int IM2COL_THREAD_FUNCT7 = `IM2COL_FUNCT7;
localparam int IM2COL_NUM_CASES = 8;
localparam int IM2COL_TIMEOUT_CYCLES = 100000;
