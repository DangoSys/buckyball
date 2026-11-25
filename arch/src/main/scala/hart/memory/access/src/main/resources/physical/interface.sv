interface cpu_control_if (
    input logic clock
);
  logic reset;
  clocking cb @(posedge clock);
    default input #1step output #0;
    output reset;
  endclocking
endinterface
