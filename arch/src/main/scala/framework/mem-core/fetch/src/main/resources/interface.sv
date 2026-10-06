interface fetch_control_if (
    input logic clock
);
  logic reset, redirect_valid, flush;
  logic [63:0] reset_vector, redirect_pc, satp, npc;
  logic [1:0] privilege;
  logic sum, mxr;
  clocking cb @(posedge clock);
    default input #1step output #0;
    input npc;
    output reset, redirect_valid, redirect_pc, flush, privilege, satp, mxr;
    output context_sum = sum;
  endclocking
endinterface
