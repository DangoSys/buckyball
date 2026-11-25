`include "parameters.svh"
package prm_types;
  localparam int BANKS = `PRM_BANKS, ROWS = `PRM_ROWS, WIDTH = `PRM_WIDTH, MASKS = `PRM_MASKS;
  typedef struct packed {
    logic read_valid, read_ready, read_result_valid, read_result_ready;
    logic write_valid, write_ready, write_result_valid, write_result_ready, write_ok;
    logic [15:0] read_addr, write_addr;
    logic [WIDTH-1:0] read_data, write_data;
    logic [MASKS-1:0] write_mask;
    logic clear, clearing, bind_generation;
    logic [31:0] hash;
  } bank_signals;
endpackage
