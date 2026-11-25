// Mirrors Mxint8mmCmd of casegen/src/cases.rs.
typedef struct {
  int unsigned rob_id;
  int unsigned iter;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned a_bank;
  int unsigned b_bank;
  int unsigned c_bank;
  int unsigned read_bank;
  int unsigned a_group;
  int unsigned b_group;
  int unsigned c_group;
  int unsigned update;
  int unsigned features;
  int unsigned stall;
} mxint8mm_cmd_dpi_t;

import "DPI-C" function int unsigned mxint8mm_num_cases();
import "DPI-C" function int unsigned mxint8mm_case_load(
  input int unsigned index, input int unsigned bank_entries, input int unsigned rob_entries
);
import "DPI-C" function void mxint8mm_case_cmd(
  input int unsigned step, output mxint8mm_cmd_dpi_t cmd
);
import "DPI-C" function int unsigned mxint8mm_case_preload_count(input int unsigned step);
import "DPI-C" function void mxint8mm_case_preload(
  input int unsigned step,
  input int unsigned i,
  output int unsigned bank,
  output int unsigned group,
  output int unsigned addr,
  output longint unsigned lo,
  output longint unsigned hi
);
import "DPI-C" function int unsigned mxint8mm_case_write_count(input int unsigned step);
import "DPI-C" function void mxint8mm_case_write(
  input int unsigned step,
  input int unsigned i,
  output int unsigned addr,
  output longint unsigned lo,
  output longint unsigned hi
);

`ifndef MXINT8MM_FUNCT7
`error "MXINT8MM_FUNCT7 must be provided by the Ball's instructions"
`endif
`ifndef MXINT8MM_ROWS
`error "MXINT8MM_ROWS must be provided by verify/parameters.toml"
`endif
`ifndef MXINT8MM_COLS
`error "MXINT8MM_COLS must be provided by verify/parameters.toml"
`endif
`ifndef MXINT8MM_LANES
`error "MXINT8MM_LANES must be provided by verify/parameters.toml"
`endif
import "DPI-C" function longint unsigned mxint8mm_bemu_cycles(
  input longint unsigned iter,
  input longint unsigned rs2,
  input longint unsigned rows,
  input longint unsigned cols,
  input longint unsigned lanes
);
localparam int MXINT8MM_TIMEOUT_CYCLES = 200000;
// Cycles the response is held back on a stalled step.
localparam int MXINT8MM_STALL_CYCLES = 40;
