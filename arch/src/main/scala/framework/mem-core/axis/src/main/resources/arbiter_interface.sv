interface arbiter_if(input logic clock);
  logic        reset;
  logic [ 1:0] in_valid;
  logic [ 1:0] in_ready;
  logic [31:0] in_data[2];
  logic [ 3:0] in_keep[2];
  logic [ 1:0] in_last;
  logic        out_valid;
  logic        out_ready;
  logic [31:0] out_data;
  logic [ 3:0] out_keep;
  logic        out_last;

  assert property (@(posedge clock) disable iff (reset) $onehot0(in_ready));
  assert property (@(posedge clock) disable iff (reset)
    out_valid && !out_ready |=> out_valid && $stable({out_data, out_keep, out_last}));

  cover property (@(posedge clock) disable iff (reset) &in_valid);
  cover property (@(posedge clock) disable iff (reset) in_valid[0] && in_ready[0]);
  cover property (@(posedge clock) disable iff (reset) in_valid[1] && in_ready[1]);
endinterface
