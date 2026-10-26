`include "parameters.svh"
interface admission_if (
    input logic clock
);
  logic reset = 1;
  logic [`AD_INPUT_BITS-1:0] inputs = '0;
  wire [`AD_OUTPUT_BITS-1:0] outputs;
endinterface
