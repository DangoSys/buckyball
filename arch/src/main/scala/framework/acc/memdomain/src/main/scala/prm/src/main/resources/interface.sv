interface bus_if (
    input logic clock
);
  import prm_types::*;
  logic reset = 1;
  bank_signals ports[BANKS];
  for (genvar b = 0; b < BANKS; b++) begin
    assert property (@(posedge clock) disable iff (reset)
      ports[b].read_result_valid && !ports[b].read_result_ready |=>
      ports[b].read_result_valid && $stable(
        ports[b].read_data
    ))
    else $fatal(1, "PRM read response changed under backpressure");
    assert property (@(posedge clock) disable iff (reset)
      ports[b].write_result_valid && !ports[b].write_result_ready |=>
      ports[b].write_result_valid && $stable(
        ports[b].write_ok
    ))
    else $fatal(1, "PRM write response changed under backpressure");
  end
endinterface
