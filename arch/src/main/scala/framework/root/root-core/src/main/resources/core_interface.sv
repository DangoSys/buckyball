interface core_control_if (
    input logic clock
);
  logic reset;
  logic timer_irq, software_irq, external_irq;
  logic load_waiting, amo_completed, store_completed;
  logic [2:0] target_access;
  logic cancelled, retired, trapped;
  logic [63:0] retired_pc, trap_cause, trap_value, trap_pc;
  logic [2:0] outstanding;
  clocking cb @(posedge clock);
    default input #1step output #0;
    output reset, timer_irq, software_irq, external_irq;
    input load_waiting, amo_completed, store_completed, target_access;
    input cancelled, outstanding, retired, retired_pc, trapped, trap_cause, trap_value, trap_pc;
  endclocking
endinterface
