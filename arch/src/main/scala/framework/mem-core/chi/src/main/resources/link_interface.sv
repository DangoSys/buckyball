interface chi_link_if (
    input logic clock
);
  logic reset, active, valid, ready;
  logic [31:0] data;
  logic out_valid, out_ready;
  logic [31:0] out_data;
  logic [ 2:0] credits;
endinterface
