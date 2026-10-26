`include "parameters.svh"
interface boot_if (
    input logic clock
);
  logic reset = 1;
  logic [`BOOT_INPUT_BITS-1:0] inputs;
  logic [`BOOT_OUTPUT_BITS-1:0] outputs;
  assert property (@(posedge clock) disable iff (reset)
    outputs[`BOOT_OUT_CMD_VALID_OFFSET] && !inputs[`BOOT_IN_CMD_READY_OFFSET] |=>
    outputs[`BOOT_OUT_CMD_VALID_OFFSET] && $stable(
      outputs[`BOOT_OUT_PACKET_OFFSET+:`BOOT_PACKET_BITS]
  ))
  else $fatal(1, "Boot output changed under backpressure");
endinterface
