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
} mxint8quant_cmd_dpi_t;

import "DPI-C" function int unsigned mxint8quant_num_cases();
import "DPI-C" function int unsigned mxint8quant_num_events();
import "DPI-C" function int unsigned mxint8quant_case_load(input int unsigned index);
import "DPI-C" function void mxint8quant_case_cmd(input int unsigned step,
                                                  output mxint8quant_cmd_dpi_t cmd);
import "DPI-C" function int unsigned mxint8quant_case_hold(input int unsigned step);
import "DPI-C" function int unsigned mxint8quant_case_event(input int unsigned step,
                                                            input int unsigned event_index);
import "DPI-C" function int unsigned mxint8quant_case_preload_count(input int unsigned step);
import "DPI-C" function int unsigned mxint8quant_case_preload_addr(input int unsigned step,
                                                                   input int unsigned i);
import "DPI-C" function longint unsigned mxint8quant_case_preload_lo(input int unsigned step,
                                                                     input int unsigned i);
import "DPI-C" function longint unsigned mxint8quant_case_preload_hi(input int unsigned step,
                                                                     input int unsigned i);
import "DPI-C" function int unsigned mxint8quant_case_write_count(input int unsigned step);
import "DPI-C" function int unsigned mxint8quant_case_write_addr(input int unsigned step,
                                                                 input int unsigned i);
import "DPI-C" function int unsigned mxint8quant_case_write_kind(input int unsigned step,
                                                                 input int unsigned i);
import "DPI-C" function longint unsigned mxint8quant_case_write_lo(input int unsigned step,
                                                                   input int unsigned i);
import "DPI-C" function longint unsigned mxint8quant_case_write_hi(input int unsigned step,
                                                                   input int unsigned i);
import "DPI-C" function longint unsigned mxint8quant_bemu_cycles(input longint unsigned iter,
                                                                input longint unsigned rs2);

`ifndef MXINT8QUANT_FUNCT7
`error "MXINT8QUANT_FUNCT7 must be provided by the selected ballISA"
`endif

localparam int MXINT8QUANT_FUNCT7_VALUE = `MXINT8QUANT_FUNCT7;
localparam int MXINT8QUANT_TIMEOUT_CYCLES = 100000;
