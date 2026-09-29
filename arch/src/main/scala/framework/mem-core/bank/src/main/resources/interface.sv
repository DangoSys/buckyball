interface bank_if (
    input logic clock
);
  logic reset;
  logic valid;
  logic ready;
  logic [3:0] addr;
  logic write;
  logic [31:0] data;
  logic [3:0] mask;
  logic [7:0] tag;
  logic error;

  assert property (@(posedge clock) disable iff (reset) valid && !ready |=> valid && $stable(
      {data, tag, error}
  ));
  cover property (@(posedge clock) disable iff (reset) valid && !ready ##1 valid && ready);
endinterface
