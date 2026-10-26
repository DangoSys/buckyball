`include "parameters.svh"
interface privileged_if (
    input logic clock
);
  logic reset = 1;
  logic [`PRIV_INPUT_BITS-1:0] inputs;
  logic [`PRIV_OUTPUT_BITS-1:0] outputs;
endinterface
