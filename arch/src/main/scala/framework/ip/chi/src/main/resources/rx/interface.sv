interface link_if (
    input logic clock
);
  import link_types::*;
  logic reset = 1;
  link_signals pins;
  assert property (@(posedge clock) disable iff (reset)
    pins.output_valid && !pins.output_ready |=> pins.output_valid && $stable(
      pins.output_data
  ))
  else $fatal(1, "CHI RX output changed under backpressure");
endinterface
