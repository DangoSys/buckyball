`include "preparation_parameters.svh"
interface preparation_if (
    input logic clock
);
  logic reset = 1;
  logic [`PREP_INPUT_BITS-1:0] source = '0, backend = '0;
  wire [ `PREP_INPUT_BITS-1:0] inputs = source | backend;
  wire [`PREP_OUTPUT_BITS-1:0] outputs;
endinterface
