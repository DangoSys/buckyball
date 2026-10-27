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
} mxfp2int_cmd_dpi_t;

import "DPI-C" function int unsigned mxfp2int_num_cases();
import "DPI-C" function int unsigned mxfp2int_case_load(input int unsigned index);
import "DPI-C" function void mxfp2int_case_cmd(input int unsigned step, output mxfp2int_cmd_dpi_t cmd);
import "DPI-C" function int unsigned mxfp2int_case_preload_count(input int unsigned step);
import "DPI-C" function int unsigned mxfp2int_case_preload_bank(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxfp2int_case_preload_group(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxfp2int_case_preload_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxfp2int_case_preload_lo(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxfp2int_case_preload_hi(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxfp2int_case_read_count(input int unsigned step, input int unsigned port);
import "DPI-C" function int unsigned mxfp2int_case_read_addr(input int unsigned step, input int unsigned port, input int unsigned i);
import "DPI-C" function int unsigned mxfp2int_case_write_count(input int unsigned step);
import "DPI-C" function int unsigned mxfp2int_case_write_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxfp2int_case_write_mask(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxfp2int_case_write_lo(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxfp2int_case_write_hi(input int unsigned step, input int unsigned i);

`ifndef MXFP2INT_FUNCT7
`error "MXFP2INT_FUNCT7 must be provided by the selected ballISA"
`endif

localparam int MXFP2INT_THREAD_FUNCT7 = `MXFP2INT_FUNCT7;
localparam int MXFP2INT_TIMEOUT_CYCLES = 100000;
