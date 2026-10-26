`include "parameters.svh"
interface rob_if (
    input logic clock
);
  logic reset = 1;
  logic [`ROB_INPUT_BITS-1:0] inputs;
  logic [`ROB_OUTPUT_BITS-1:0] outputs;
endinterface
