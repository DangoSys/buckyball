`include "parameters.svh"
interface cmo_if (
    input logic clock,
    input logic reset
);
  logic valid = 0, ready, response_valid, response_ready = 1, ok;
  logic [`CMO_ID_BITS-1:0] tag = 0, response_tag;
  logic [`CHI_ADDRESS_BITS-1:0] first_line = 0, last_line = 0;
  logic [1:0] operation = 0;
  logic older_drained = 1;
  assert property (@(posedge clock) disable iff (reset) valid && !ready |=> valid && $stable(
      {tag, first_line, last_line, operation}
  ))
  else $fatal(1, "Maintenance request changed under backpressure");
  assert property(@(posedge clock) disable iff(reset)response_valid&&!response_ready|=>response_valid&&$stable(
      {response_tag, ok}
  ))
  else $fatal(1, "Maintenance response changed under backpressure");
endinterface
