`include "coherence_config.svh"
interface home_transport_if (
    input logic clock,
    input logic reset
);
  logic [5:0] expected_valid, actual_valid;
  logic [`COH_DAT_WIDTH-1:0] expected_bits[6], actual_bits[6];
  int expected_node[6], actual_node[6];
endinterface
