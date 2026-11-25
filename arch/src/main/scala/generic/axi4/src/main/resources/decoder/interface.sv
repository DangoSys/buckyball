interface decoder_if (
    input logic clock
);
  import axi_types::*;
  logic reset = 1;
  port_signals ports[3];
endinterface
