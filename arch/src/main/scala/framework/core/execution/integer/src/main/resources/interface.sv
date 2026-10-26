interface port_if (
    input logic clock,
    input logic reset
);
  logic sample;
  logic [63:0] a;
  logic [63:0] rhs;
  logic [4:0] fn;
  logic dw;
  logic [63:0] value;
  logic [63:0] adder;
  logic comparison;
endinterface
