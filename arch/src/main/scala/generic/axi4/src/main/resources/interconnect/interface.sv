interface bus_if (
    input logic clock,
    input logic reset
);
  import axi_types::*;
  port_signals ports[M+1];
  logic [$clog2(2*`AXI_SLOTS+1)-1:0] outstanding;
  for (genvar p = 0; p <= M; p++) begin
    assert property (@(posedge clock) disable iff (reset)
      ports[p].aw_valid && !ports[p].aw_ready |=> ports[p].aw_valid && $stable(
        ports[p].aw
    ))
    else $fatal(1, "AXI AW changed while stalled");
    assert property (@(posedge clock) disable iff (reset)
      ports[p].ar_valid && !ports[p].ar_ready |=> ports[p].ar_valid && $stable(
        ports[p].ar
    ))
    else $fatal(1, "AXI AR changed while stalled");
    assert property (@(posedge clock) disable iff (reset)
      ports[p].w_valid && !ports[p].w_ready |=> ports[p].w_valid && $stable(
        ports[p].w
    ))
    else $fatal(1, "AXI W changed while stalled");
    assert property (@(posedge clock) disable iff (reset)
      ports[p].r_valid && !ports[p].r_ready |=> ports[p].r_valid && $stable(
        ports[p].r
    ))
    else $fatal(1, "AXI R changed while stalled");
    assert property (@(posedge clock) disable iff (reset)
      ports[p].b_valid && !ports[p].b_ready |=> ports[p].b_valid && $stable(
        ports[p].b
    ))
    else $fatal(1, "AXI B changed while stalled");
  end
endinterface
