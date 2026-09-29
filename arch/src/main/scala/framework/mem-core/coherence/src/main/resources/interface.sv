`include "coherence_config.svh"
interface coherence_control_if (
    input logic clock
);
  logic reset;
  logic [`COH_OUTSTANDING_BITS-1:0] outstanding;
  clocking cb @(posedge clock);
    default input #1step output #0;
    input outstanding;
    output reset;
  endclocking
endinterface
