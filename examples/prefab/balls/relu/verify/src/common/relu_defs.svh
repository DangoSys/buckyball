typedef struct {
  int unsigned iter;
  int unsigned stride;
  int unsigned bank;
  int unsigned group_base;
  int unsigned group;
  int unsigned cols;
  int unsigned rob_id;
  int unsigned num_words;
} relu_cmd_dpi_t;

import "DPI-C" function int unsigned relu_bank_entries();
import "DPI-C" function int relu_case_load(input int unsigned index);
import "DPI-C" function void relu_case_cmd(output relu_cmd_dpi_t cmd);
import "DPI-C" function int unsigned relu_case_row(input int unsigned word_index);
import "DPI-C" function longint unsigned relu_case_src_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned relu_case_src_word_hi(input int unsigned word_index);
import "DPI-C" function longint unsigned relu_case_dst_word_lo(input int unsigned word_index);
import "DPI-C" function longint unsigned relu_case_dst_word_hi(input int unsigned word_index);

`ifndef RELU_FUNCT7
`error "RELU_FUNCT7 must be provided by the selected ballISA"
`endif

localparam int RELU_THREAD_FUNCT7 = `RELU_FUNCT7;
localparam int RELU_NUM_CASES = 6;
localparam int RELU_TIMEOUT_CYCLES = 100000;
