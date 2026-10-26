interface port_if (
    input logic clock,
    input logic reset
);
  logic sample;
  logic [4:0] read0;
  logic [4:0] read1;
  logic write;
  logic [4:0] address;
  logic [63:0] data;
  logic [63:0] data0;
  logic [63:0] data1;
endinterface
