`include "parameters.svh"
interface bank_if (
    input logic clock
);
  logic reset = 1;
  logic [`SB_INPUT_BITS-1:0] inputs;
  logic [`SB_OUTPUT_BITS-1:0] outputs;
endinterface
