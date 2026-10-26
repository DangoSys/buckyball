`include "parameters.svh"
interface global_if (
    input logic clock
);
  logic reset;
  logic [`GROB_INPUT_BITS-1:0] inputs;
  logic [`GROB_OUTPUT_BITS-1:0] outputs;
  assert property (@(posedge clock) disable iff (reset)
    outputs[`GROB_OUT_ISSUE_VALID_OFFSET] && !inputs[`GROB_IN_ISSUE_READY_OFFSET] |=>
    outputs[`GROB_OUT_ISSUE_VALID_OFFSET] && $stable(
      outputs[`GROB_ISSUE_OFFSET+:`GROB_ISSUE_BITS]
  ))
  else $fatal(1, "Global ROB issue payload changed under backpressure");
endinterface
