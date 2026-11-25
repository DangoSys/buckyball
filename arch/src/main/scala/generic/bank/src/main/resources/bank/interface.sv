`include "parameters.svh"
interface port_if (
    input logic clock
);
  logic reset;
  logic valid;
  logic ready;
  logic [`BANK_ROW_BITS-1:0] addr;
  logic write;
  logic [31:0] data;
  logic [3:0] mask;
  logic [`BANK_TAG_BITS-1:0] tag;
  logic error;

  assert property (@(posedge clock) disable iff (reset) valid && !ready |=> valid && $stable(
      {addr, write, data, mask, tag, error}
  ))
  else $fatal(1, "Bank payload changed under backpressure");
  cover property (@(posedge clock) disable iff (reset) valid && !ready ##1 valid && ready);
endinterface
