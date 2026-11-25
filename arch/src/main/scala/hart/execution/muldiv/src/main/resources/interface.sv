interface port_if (
    input logic clock,
    input logic reset
);
  logic request_valid, request_ready, response_valid, response_ready;
  logic [63:0] a;
  logic [63:0] rhs;
  logic [2:0] operation;
  logic word;
  int unsigned stall;
  logic cancel;
  logic [63:0] value;
  assert property (@(posedge clock) disable iff (reset)
    request_valid && !request_ready |=> request_valid && $stable(
      {a, rhs, operation, word}
  ))
  else $fatal(1, "MulDiv request changed under backpressure");
  assert property (@(posedge clock) disable iff (reset)
    response_valid && !response_ready |=> response_valid && $stable(
      value
  ))
  else $fatal(1, "MulDiv result changed under backpressure");
endinterface
