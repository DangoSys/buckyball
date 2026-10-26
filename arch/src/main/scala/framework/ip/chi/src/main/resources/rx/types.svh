`include "parameters.svh"
package link_types;
  localparam int WIDTH = `CHI_LINK_BITS, DEPTH = `CHI_CREDITS;
  typedef struct packed {
    logic active, input_valid, input_ready, output_valid, output_ready, pend, credit;
    logic [WIDTH-1:0] input_data, output_data;
    logic [`CHI_CREDIT_BITS-1:0] credits;
  } link_signals;
endpackage
