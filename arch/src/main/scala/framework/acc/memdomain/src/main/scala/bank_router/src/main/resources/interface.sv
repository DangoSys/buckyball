`include "parameters.svh"
interface router_if (
    input logic clock
);
  logic reset = 1;
  logic [`BR_INPUT_BITS-1:0] source = '0, backend = '0;
  wire [ `BR_INPUT_BITS-1:0] inputs = source | backend;
  wire [`BR_OUTPUT_BITS-1:0] outputs;
endinterface
