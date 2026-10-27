typedef struct {
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned rob_id;
} trace_cmd_dpi_t;

import "DPI-C" function int unsigned trace_case_load(input int unsigned index);
import "DPI-C" function void trace_case_cmd(input int unsigned index, output trace_cmd_dpi_t cmd);
import "DPI-C" function int trace_case_records();

`ifndef BDB_COUNTER_FUNCT7
`error "BDB_COUNTER_FUNCT7 must be provided by the selected ballISA"
`endif

localparam int BDB_COUNTER_THREAD_FUNCT7 = `BDB_COUNTER_FUNCT7;
localparam int TRACE_NUM_CASES = 4;
localparam int TRACE_TIMEOUT_CYCLES = 2000;
