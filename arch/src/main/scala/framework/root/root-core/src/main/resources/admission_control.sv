`include "admission_config.svh"
interface admission_control_if (
    input logic clock
);
  logic reset, retired, trapped, cancelled;
  logic [63:0] retired_pc, trap_pc, trap_cause, trap_value;
  logic [`ADMIT_MEMORY_OUTSTANDING_BITS-1:0] outstanding;
  logic [`ADMIT_OUTSTANDING_BITS-1:0] admission_outstanding;
  logic [`ADMIT_QUERY_WIDTH-1:0] query;
  logic cpu_allow, accelerator_irq;
  logic block_maintenance_response, maintenance_response_pending;
  bit start = 0, finished = 0, hold_complete = 1;
endinterface
