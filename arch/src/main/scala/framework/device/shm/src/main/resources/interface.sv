interface storage_if (
    input logic clock
);
  logic reset, valid, ready, write;
  logic [ 63:0] address;
  logic [127:0] data;
  logic [ 15:0] mask;
  logic response_valid, response_ready, error;
  logic [127:0] result;
endinterface
