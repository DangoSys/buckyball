interface arbiter_if (
    input logic clock
);
  logic                         reset;
  logic [                  1:0] in_valid;
  logic [                  1:0] in_ready;
  logic [  `AXIS_DATA_BITS-1:0] in_data   [2];
  logic [`AXIS_DATA_BITS/8-1:0] in_keep   [2];
  logic [                  1:0] in_last;
  logic                         out_valid;
  logic                         out_ready;
  logic [  `AXIS_DATA_BITS-1:0] out_data;
  logic [`AXIS_DATA_BITS/8-1:0] out_keep;
  logic                         out_last;

  assert property (@(posedge clock) disable iff (reset) $onehot0(in_ready))
  else $fatal(1, "Arbiter accepted more than one input");
  assert property (@(posedge clock) disable iff (reset)
    out_valid && !out_ready |=> out_valid && $stable(
      {out_data, out_keep, out_last}
  ))
  else $fatal(1, "Arbiter output changed under backpressure");

  for (genvar i = 0; i < 2; i++) begin
    assert property (@(posedge clock) disable iff (reset)
      in_valid[i] && !in_ready[i] |=> in_valid[i] &&
      $stable(
        {in_data[i], in_keep[i], in_last[i]}
    ))
    else $fatal(1, "Source changed an unaccepted input beat");
  end

  cover property (@(posedge clock) disable iff (reset) &in_valid);
  cover property (@(posedge clock) disable iff (reset) in_valid[0] && in_ready[0]);
  cover property (@(posedge clock) disable iff (reset) in_valid[1] && in_ready[1]);
endinterface
