interface maintenance_control_if (
    input logic clock
);
  logic reset, drained, idle;
  clocking cb @(posedge clock);
    default input #1step output #0;
    output reset, drained;
    input idle;
  endclocking
endinterface
