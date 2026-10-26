`include "parameters.svh"
interface tomem_if (
    input logic clock
);
  logic reset = 1;
  logic [`TM_INPUT_BITS-1:0] source = '0, backend = '0;
  wire [ `TM_INPUT_BITS-1:0] inputs = source | backend;
  wire [`TM_OUTPUT_BITS-1:0] outputs;
endinterface
