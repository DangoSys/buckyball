typedef struct {
  int unsigned bid;
  int unsigned iter;
  int unsigned gate_bank;
  int unsigned input_bank;
  int unsigned output_bank;
  int unsigned op1_col;
  int unsigned op2_col;
  int unsigned wr_col;
  int unsigned gate_row;
  int unsigned rob_id;
  int unsigned rs1_lo;
  int unsigned rs1_hi;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned num_gate_words;
  int unsigned num_input_words;
  int unsigned num_dst_words;
} int8mul_cmd_dpi_t;

import "DPI-C" function int int8mul_case_load(
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void int8mul_case_cmd(output int8mul_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned int8mul_case_gate_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_gate_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_input_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_input_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned int8mul_case_dst_word_hi(input int unsigned word_index);

`ifndef INT8MUL_FUNCT7
`error "INT8MUL_FUNCT7 must be provided by the selected Core ballISA"
`endif

localparam int INT8MUL_CORE_FUNCT7 = `INT8MUL_FUNCT7;
localparam int INT8MUL_MAX_ROWS = 64;
localparam int INT8MUL_NUM_CASES = 2;
localparam int INT8MUL_TIMEOUT_CYCLES = 20000;

function automatic int unsigned int8mul_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
