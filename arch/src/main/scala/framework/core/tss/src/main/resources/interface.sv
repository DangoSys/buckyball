`include "parameters.svh"
interface tss_if (
    input logic clock
);
  localparam int PORT_OFFSETS[`TSS_PORTS][13] = `TSS_PORT_OFFSETS;
  logic reset = 1;
  logic [`TSS_INPUT_BITS-1:0] inputs = '0;
  logic [`TSS_OUTPUT_BITS-1:0] outputs;
  for (genvar p = 0; p < `TSS_PORTS; p++) begin
    assert property (@(posedge clock) disable iff(reset || (p<`TSS_CLIENTS && inputs[PORT_OFFSETS[p][12]]))
      outputs[PORT_OFFSETS[p][7]] && !inputs[PORT_OFFSETS[p][10]] |=> outputs[PORT_OFFSETS[p][7]] &&
      $stable(
        {outputs[PORT_OFFSETS[p][8]+:`TSS_DATA_BITS], outputs[PORT_OFFSETS[p][9]]}
    ))
    else $fatal(1, "TSS port response changed under backpressure");
  end
endinterface
