`include "virtual_parameters.svh"
interface virtual_if (
    input logic clock
);
  logic reset = 1;
  logic [`VA_INPUT_BITS-1:0] source = '0, backend = '0;
  wire [ `VA_INPUT_BITS-1:0] inputs = source | backend;
  wire [`VA_OUTPUT_BITS-1:0] outputs;
endinterface
