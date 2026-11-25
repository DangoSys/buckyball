interface link_if (
    input logic clock
);
  import link_types::*;
  logic reset = 1;
  link_signals pins;
endinterface
