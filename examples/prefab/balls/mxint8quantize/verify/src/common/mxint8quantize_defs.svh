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
} mxint8quantize_cmd_dpi_t;

import "DPI-C" function int unsigned mxint8quantize_num_cases();
import "DPI-C" function int unsigned mxint8quantize_bank_entries();
import "DPI-C" function int unsigned mxint8quantize_case_load(input int unsigned index);
import "DPI-C" function void mxint8quantize_case_cmd(input int unsigned step, output mxint8quantize_cmd_dpi_t cmd);
import "DPI-C" function int unsigned mxint8quantize_case_preload_count(input int unsigned step);
import "DPI-C" function int unsigned mxint8quantize_case_preload_bank(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxint8quantize_case_preload_group(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxint8quantize_case_preload_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxint8quantize_case_preload_lo(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxint8quantize_case_preload_hi(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxint8quantize_case_read_count(input int unsigned step);
import "DPI-C" function int unsigned mxint8quantize_case_read_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxint8quantize_case_write_count(input int unsigned step);
import "DPI-C" function int unsigned mxint8quantize_case_write_addr(input int unsigned step, input int unsigned i);
import "DPI-C" function int unsigned mxint8quantize_case_write_mask(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxint8quantize_case_write_lo(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxint8quantize_case_write_hi(input int unsigned step, input int unsigned i);
import "DPI-C" function longint unsigned mxint8quantize_bemu_cycles(input longint unsigned iter,
                                                                  input longint unsigned rs2);
import "DPI-C" function int unsigned mxint8quantize_num_events();
import "DPI-C" function int unsigned mxint8quantize_case_event(input int unsigned step, input int unsigned event_id);

`ifndef MXINT8QUANT_FUNCT7
`error "MXINT8QUANT_FUNCT7 must be provided from the Ball's configs/default.toml"
`endif

localparam int MXINT8QUANT_FUNCT7_VALUE = `MXINT8QUANT_FUNCT7;
localparam int MXINT8QUANT_TIMEOUT_CYCLES = 200000;
