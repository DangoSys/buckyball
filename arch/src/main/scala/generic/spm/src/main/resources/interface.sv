`include "parameters.svh"
interface port_if (
    input logic clock,
    input logic reset
);
  logic request_valid, request_ready, response_valid, response_ready, busy;
  logic [63:0] address;
  logic [$clog2($clog2(`SPM_BITS/8)+2)-1:0] size;
  logic write;
  logic [`SPM_BITS-1:0] data;
  logic [`SPM_BITS/8-1:0] mask;
  logic read_only;
  int unsigned stall;
  logic cancel;
  logic [`SPM_BITS-1:0] value;
  logic error;
  assert property (@(posedge clock) disable iff (reset)
    request_valid && !request_ready |=> request_valid && $stable(
      {address, size, write, data, mask, read_only}
  ))
  else $fatal(1, "SPM request changed under backpressure");
  assert property (@(posedge clock) disable iff (reset)
    response_valid && !response_ready |=> response_valid && $stable(
      {value, error}
  ))
  else $fatal(1, "SPM response changed under backpressure");
endinterface
