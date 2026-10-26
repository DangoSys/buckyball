`include "parameters.svh"
interface mmu_if (
    input logic clock
);
  logic reset;
  logic [`MMU_INPUT_BITS-1:0] inputs = '0;
  logic [`MMU_OUTPUT_BITS-1:0] outputs;
  assert property (@(posedge clock) disable iff (reset)
    outputs[`MMU_OUT_ACCESS_VALID_OFFSET] && !inputs[`MMU_IN_ACCESS_READY_OFFSET] |=>
    outputs[`MMU_OUT_ACCESS_VALID_OFFSET] && $stable(
      outputs[`MMU_OUT_ACCESS_BITS_ADDR_OFFSET+:`MMU_OUT_ACCESS_BITS_ADDR_WIDTH]
  ))
  else $fatal(1, "Walker PTE address changed under backpressure");
  assert property (@(posedge clock) disable iff (reset)
    outputs[`MMU_OUT_RESP_VALID_OFFSET] && !inputs[`MMU_IN_RESP_READY_OFFSET] |=>
    outputs[`MMU_OUT_RESP_VALID_OFFSET] && $stable(
      outputs[`MMU_RESP_OFFSET+:`MMU_RESP_BITS]
  ))
  else $fatal(1, "Walker translation changed under backpressure");
endinterface
