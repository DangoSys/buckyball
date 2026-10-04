#include "task_dma.h"
void task_dma_kernel(void *);
volatile uint8_t task_dma_input[DMA_BYTES] __attribute__((aligned(64)));
volatile uint8_t task_dma_output[DMA_BYTES] __attribute__((aligned(64)));
volatile uint32_t task_dma_result;
static uint8_t stack[4096] __attribute__((aligned(16)));
static uint8_t tls[64] __attribute__((aligned(16)));
static struct dma_argument argument;
int main(void) {
  core_id_t id = bb_get_core_id();
  if (id.core)
    buckyball_task_worker();
  if (id.tile != TASK_TILE || bb_task_worker_count() != TASK_WORKERS ||
      !bb_task_core_signature(0))
    return 1;
  for (unsigned i = 0; i < DMA_BYTES; i++) {
    task_dma_input[i] = (uint8_t)(17 + i * 3);
    task_dma_output[i] = 0xa5;
  }
  argument = (struct dma_argument){(uintptr_t)task_dma_input,
                                   (uintptr_t)task_dma_output,
                                   (uintptr_t)&task_dma_result, 1};
  task_tls(tls);
  if (submit_task(0, task_dma_kernel, &argument, stack + sizeof(stack), tls))
    return 2;
  uintptr_t status;
  do {
    status = bb_task_status(0);
  } while (!status);
  if (status != 1 || task_dma_result != 1)
    return 3;
  for (unsigned i = 0; i < DMA_BYTES; i++)
    if (task_dma_output[i] != (uint8_t)(17 + i * 3))
      return 4;
  return 0;
}
