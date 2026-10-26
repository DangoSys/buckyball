`include "parameters.svh"
interface tbus_if (
    input logic clock
);
  logic reset = 1;
  logic [`TR_INPUT_BITS-1:0] source = '0, backend = '0;
  wire [ `TR_INPUT_BITS-1:0] inputs = source | backend;
  wire [`TR_OUTPUT_BITS-1:0] outputs;
endinterface
