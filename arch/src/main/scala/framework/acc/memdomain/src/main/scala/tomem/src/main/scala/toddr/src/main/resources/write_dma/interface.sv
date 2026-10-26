interface dma_if (
    input logic clock
);
  logic reset = 1;
  `include "write_dma_signals.svh"
  `include "write_dma_clocking.svh"
  task initialize();
    `include "write_dma_init.svh"
  endtask
endinterface
