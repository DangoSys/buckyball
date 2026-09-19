interface axis_if(input logic clock);
  logic        reset;
  logic        tvalid;
  logic        tready;
  logic [31:0] tdata;
  logic [ 3:0] tkeep;
  logic        tlast;

  property payload_stable_during_stall;
    @(posedge clock) disable iff (reset)
      tvalid && !tready |=> tvalid && $stable({tdata, tkeep, tlast});
  endproperty

  assert property (payload_stable_during_stall);
  cover property (@(posedge clock) disable iff (reset) tvalid && !tready ##1 tvalid && tready);
endinterface
