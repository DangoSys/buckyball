typedef struct {
  int unsigned bid;
  int unsigned ws;
  int unsigned m;
  int unsigned n;
  int unsigned k;
  int unsigned op1_bank;
  int unsigned op2_bank;
  int unsigned wr_bank;
  int unsigned rob_id;
  int unsigned rs1_lo;
  int unsigned rs1_hi;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned num_a_words;
  int unsigned num_b_words;
  int unsigned num_writes;
} matrix_cmd_dpi_t;

import "DPI-C" function int smatmul_case_load(
  input int unsigned seed,
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void smatmul_case_cmd(output matrix_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned smatmul_case_a_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned smatmul_case_a_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned smatmul_case_b_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned smatmul_case_b_word_hi(input int unsigned word_index);
import "DPI-C" function int unsigned smatmul_case_num_writes();
import "DPI-C" function int unsigned smatmul_case_write_group(input int unsigned i);
import "DPI-C" function int unsigned smatmul_case_write_addr(input int unsigned i);
import "DPI-C" function longint unsigned smatmul_case_write_data_lo(input int unsigned i);
import "DPI-C" function longint unsigned smatmul_case_write_data_hi(input int unsigned i);
import "DPI-C" function int unsigned smatmul_case_write_mask(input int unsigned i);

`ifndef SMATMUL_OS_FUNCT7
`error "SMATMUL_OS_FUNCT7 must be provided by the selected Core ballISA"
`endif
`ifndef SMATMUL_WS_FUNCT7
`error "SMATMUL_WS_FUNCT7 must be provided by the selected Core ballISA"
`endif
localparam int SMATMUL_OS_CORE_FUNCT7 = `SMATMUL_OS_FUNCT7;
localparam int SMATMUL_WS_CORE_FUNCT7 = `SMATMUL_WS_FUNCT7;
localparam int MATRIX_MAX_WORDS = 32;
localparam int MATRIX_TIMEOUT_CYCLES = 100000;
localparam int MATRIX_SEED = 32'hCAFE_BABE;

function automatic int unsigned matrix_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
