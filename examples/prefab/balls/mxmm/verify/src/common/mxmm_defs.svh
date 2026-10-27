typedef struct {
  int unsigned format;
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
} mxmm_cmd_dpi_t;

import "DPI-C" function int unsigned mxmm_num_cases();
import "DPI-C" function int unsigned mxmm_case_load(input int unsigned index);
import "DPI-C" function void mxmm_case_cmd(input int unsigned step, output mxmm_cmd_dpi_t cmd);
import "DPI-C" function int unsigned mxmm_case_preload_count(input int unsigned step);
import "DPI-C" function int unsigned mxmm_case_preload_bank(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxmm_case_preload_group(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxmm_case_preload_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxmm_case_preload_lo(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxmm_case_preload_hi(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxmm_case_read_count(input int unsigned step, input int unsigned port);
import "DPI-C" function int unsigned mxmm_case_read_addr(input int unsigned step, input int unsigned port, input int unsigned i);
import "DPI-C" function int unsigned mxmm_case_write_count(input int unsigned step);
import "DPI-C" function int unsigned mxmm_case_write_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxmm_case_write_mask(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxmm_case_write_lo(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxmm_case_write_hi(input int unsigned step, input int unsigned i);

`ifndef MXMM_MXFP8_FUNCT7
`error "MXMM_MXFP8_FUNCT7 must be provided by the selected ballISA"
`endif
`ifndef MXMM_MXFP8_WINDOW_FUNCT7
`error "MXMM_MXFP8_WINDOW_FUNCT7 must be provided by the selected ballISA"
`endif
`ifndef MXMM_FMA32_FUNCT7
`error "MXMM_FMA32_FUNCT7 must be provided by the selected ballISA"
`endif
`ifndef MXMM_F32_FUNCT7
`error "MXMM_F32_FUNCT7 must be provided by the selected ballISA"
`endif

// Formats of the reference model, in the order of mxmm_case_cmd.format.
localparam int MXMM_FUNCT7[4] = '{`MXMM_MXFP8_FUNCT7, `MXMM_MXFP8_WINDOW_FUNCT7, `MXMM_FMA32_FUNCT7,
                                  `MXMM_F32_FUNCT7};
localparam int MXMM_TIMEOUT_CYCLES = 100000;
