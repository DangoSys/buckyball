`include "parameters.svh"
interface fetch_if (
    input logic clock
);
  logic reset;
  logic [`FETCH_INPUT_BITS-1:0] inputs = '0;
  logic [`FETCH_OUTPUT_BITS-1:0] outputs;
  assert property (@(posedge clock) disable iff (reset)
    outputs[`FETCH_OUT_REQUEST_VALID_OFFSET] && !inputs[`FETCH_IN_REQUEST_READY_OFFSET] |=>
    outputs[`FETCH_OUT_REQUEST_VALID_OFFSET] && $stable(
      outputs[`FETCH_REQUEST_OFFSET+:`FETCH_REQUEST_BITS]
  ))
  else $fatal(1, "Fetch word request changed under backpressure");
  assert property (@(posedge clock) disable iff (reset || inputs[`FETCH_IN_REDIRECT_VALID_OFFSET] || inputs[`FETCH_IN_FLUSH_OFFSET])
    outputs[`FETCH_OUT_PACKET_VALID_OFFSET] && !inputs[`FETCH_IN_PACKET_READY_OFFSET] |=>
    outputs[`FETCH_OUT_PACKET_VALID_OFFSET] && $stable(
      outputs[`FETCH_PACKET_OFFSET+:`FETCH_PACKET_BITS]
  ))
  else $fatal(1, "Fetch packet changed under backpressure");
endinterface
