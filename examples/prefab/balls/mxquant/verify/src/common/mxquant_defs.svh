typedef struct {
  int unsigned rob_id;
  int unsigned iter;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned op1_bank;
  int unsigned op2_bank;
  int unsigned wr_bank;
  int unsigned op1_group;
  int unsigned op2_group;
  int unsigned wr_group;
  int unsigned op1_col;
  int unsigned op2_col;
  int unsigned wr_col;
} mxquant_cmd_dpi_t;

import "DPI-C" function int unsigned mxquant_num_cases();
import "DPI-C" function int unsigned mxquant_case_load(input int unsigned index);
import "DPI-C" function void mxquant_case_cmd(input int unsigned step, output mxquant_cmd_dpi_t cmd);
import "DPI-C" function int unsigned mxquant_case_preload_count(input int unsigned step);
import "DPI-C" function int unsigned mxquant_case_preload_bank(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxquant_case_preload_group(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxquant_case_preload_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxquant_case_preload_lo(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxquant_case_preload_hi(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxquant_case_read_count(input int unsigned step);
import "DPI-C" function int unsigned mxquant_case_read_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxquant_case_write_count(input int unsigned step);
import "DPI-C" function int unsigned mxquant_case_write_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxquant_case_write_mask(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxquant_case_write_lo(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxquant_case_write_hi(input int unsigned step, input int unsigned i);

`ifndef MXQUANT_FUNCT7
`error "MXQUANT_FUNCT7 must be provided by the selected ballISA"
`endif

localparam int MXQUANT_THREAD_FUNCT7 = `MXQUANT_FUNCT7;
localparam int MXQUANT_TIMEOUT_CYCLES = 100000;
