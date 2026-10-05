interface ram_control_if (
    input logic clock
);
  logic reset;
  logic [3:0] outstanding;
  clocking sample @(posedge clock);
    default input #1step;
    input reset, outstanding;
  endclocking
endinterface
