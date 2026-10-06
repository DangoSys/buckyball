interface lsu_control_if (
    input logic clock
);
  logic reset, s1_kill, s2_kill, cancel_offer;
  logic [45:0] pc;
  logic [63:0] s1_data;
  logic nack, ordered, idle, cancelled;
  logic ma_ld, ma_st, pf_ld, pf_st, ae_ld, ae_st;
  clocking cb @(posedge clock);
    default input #1step output #0;
    output reset, s1_kill, s2_kill, pc, s1_data, cancel_offer;
    input nack, ordered, idle, cancelled, ma_ld, ma_st, pf_ld, pf_st, ae_ld, ae_st;
  endclocking
endinterface
