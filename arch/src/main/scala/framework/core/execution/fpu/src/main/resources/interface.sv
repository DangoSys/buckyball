interface port_if (
    input logic clock,
    input logic reset
);
  logic request_valid, request_ready, response_valid, response_ready;
  logic [31:0] instruction;
  logic [63:0] integer_value;
  logic [63:0] a;
  logic [63:0] rhs;
  logic [63:0] c;
  logic [2:0] rounding;
  int unsigned stall;
  logic cancel;
  logic [63:0] value;
  logic [4:0] flags;
  logic integer_result;
  logic illegal;
  assert property (@(posedge clock) disable iff (reset)
    request_valid && !request_ready |=> request_valid &&
    $stable(
      {instruction, integer_value, a, rhs, c, rounding}
  ))
  else $fatal(1, "FPU request changed under backpressure");
  assert property (@(posedge clock) disable iff (reset)
    response_valid && !response_ready |=> response_valid && $stable(
      {value, flags, integer_result, illegal}
  ))
  else $fatal(1, "FPU result changed under backpressure");
endinterface
