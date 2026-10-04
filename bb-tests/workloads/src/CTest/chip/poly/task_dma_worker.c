#include "task_dma.h"
void task_dma_kernel(void *opaque) {
  struct dma_argument *arg = opaque;
  volatile uint8_t *output = (volatile uint8_t *)arg->output;
  for (unsigned i = 0; i < DMA_BYTES; i++)
    output[i] = 0x5a;
  bb_mset(0, 1, 1, 1);
  bb_mvin(arg->input, 0, DMA_ROWS, 1);
  bb_mvout(arg->output, 0, DMA_ROWS, 1);
  for (unsigned i = 0; i < DMA_BYTES; i++)
    if (output[i] != (uint8_t)(arg->seed * 17 + i * 3))
      task_finish(1);
  *(volatile uint32_t *)arg->result = 1;
  task_finish(0);
}
