typedef struct {
  int unsigned bid;
  int unsigned iter;
  int unsigned op1_bank;
  int unsigned wr_bank;
  int unsigned op1_col;
  int unsigned wr_col;
  int unsigned rob_id;
  int unsigned rs1_lo;
  int unsigned rs1_hi;
  int unsigned rs2_lo;
  int unsigned rs2_hi;
  int unsigned input_base;
  int unsigned output_base;
  int unsigned output_stride;
  int unsigned input_side;
  int unsigned output_side;
  int unsigned kernel;
  int unsigned stride;
  int unsigned padding;
  int unsigned start_row;
  int unsigned start_col;
  int unsigned num_input_words;
  int unsigned num_dst_words;
} maxpool_cmd_dpi_t;

import "DPI-C" function int maxpool_case_load(
  input int unsigned index,
  input int unsigned bid
);
import "DPI-C" function void maxpool_case_cmd(output maxpool_cmd_dpi_t cmd);
import "DPI-C" function longint unsigned maxpool_case_input_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned maxpool_case_input_word_hi(input int unsigned word_index);
import "DPI-C" function int unsigned maxpool_case_input_addr(input int unsigned word_index);
import "DPI-C" function longint unsigned maxpool_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned maxpool_case_dst_word_hi(input int unsigned word_index);
import "DPI-C" function int unsigned maxpool_case_dst_addr(input int unsigned word_index);

`ifndef MAXPOOL_FUNCT7
`error "MAXPOOL_FUNCT7 must be provided by the selected Core ballISA"
`endif

localparam int MAXPOOL_CORE_FUNCT7 = `MAXPOOL_FUNCT7;
localparam int MAXPOOL_MAX_INPUT_WORDS = 64;
localparam int MAXPOOL_MAX_DST_WORDS = 64;
localparam int MAXPOOL_NUM_CASES = 8;
localparam int MAXPOOL_TIMEOUT_CYCLES = 20000;

function automatic int unsigned maxpool_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
