`include "preflight_config.svh"
interface prepared_map_control_if (
    input logic clock
);
  logic reset, hold_ready;
  logic [`PF_QUERY_WIDTH-1:0] query_0, query_1;
  logic [`PF_RESULT_WIDTH-1:0] result_0, result_1;
  clocking sample @(posedge clock);
    default input #1step;
    input reset, query_0, query_1, result_0, result_1;
  endclocking
endinterface
