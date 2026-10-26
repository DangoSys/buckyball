interface dma_if (
    input logic clock
);
  logic reset = 1;
  `include "read_dma_signals.svh"
  `include "read_dma_clocking.svh"
  task initialize();
    `include "read_dma_init.svh"
  endtask
endinterface
