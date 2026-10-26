interface dma_if (
    input logic clock
);
  logic reset = 1;
  `include "toddr_signals.svh"
  `include "toddr_clocking.svh"
  task initialize();
    `include "toddr_init.svh"
  endtask
endinterface
