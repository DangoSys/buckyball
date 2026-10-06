interface chi_credit_if (
    input logic clock
);
  logic reset, active, pend, valid, lcrdv;
  logic [31:0] flit;
endinterface
