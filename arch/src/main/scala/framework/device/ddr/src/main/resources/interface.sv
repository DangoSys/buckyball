interface ddr_control_if (
    input logic clock
);
  logic reset;
  clocking sample @(posedge clock);
    default input #1step;
    input reset;
  endclocking
endinterface
