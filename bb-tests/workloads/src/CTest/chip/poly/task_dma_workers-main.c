#include "task_dma.h"
void task_dma_kernel(void *);
void task_move_store_kernel(void *);
volatile uint8_t task_dma_inputs[TASK_WORKERS][DMA_BYTES]
    __attribute__((aligned(64)));
volatile uint8_t task_dma_outputs[TASK_WORKERS][DMA_BYTES]
    __attribute__((aligned(64)));
volatile uint8_t task_move_output[DMA_BYTES] __attribute__((aligned(64)));
volatile uint32_t task_dma_results[TASK_WORKERS], task_move_result;
static uint8_t stacks[TASK_WORKERS][4096] __attribute__((aligned(16)));
static uint8_t tls[TASK_WORKERS][64] __attribute__((aligned(16)));
static struct dma_argument arguments[TASK_WORKERS], moved;
int main(void) {
  core_id_t id = bb_get_core_id();
  if (id.core)
    buckyball_task_worker();
  if (id.tile != TASK_TILE || bb_task_worker_count() != TASK_WORKERS)
    return 1;
  for (unsigned rank = 0; rank < TASK_WORKERS; rank++) {
    if (!bb_task_core_signature(rank))
      return 2;
    for (unsigned i = 0; i < DMA_BYTES; i++) {
      task_dma_inputs[rank][i] = (uint8_t)((rank + 1) * 17 + i * 3);
      task_dma_outputs[rank][i] = 0xa5;
    }
    arguments[rank] = (struct dma_argument){
        (uintptr_t)task_dma_inputs[rank], (uintptr_t)task_dma_outputs[rank],
        (uintptr_t)&task_dma_results[rank], rank + 1};
    task_tls(tls[rank]);
    if (submit_task(rank, task_dma_kernel, &arguments[rank],
                    stacks[rank] + sizeof(stacks[rank]), tls[rank]))
      return 3;
  }
  for (unsigned rank = 0; rank < TASK_WORKERS; rank++) {
    uintptr_t status;
    do {
      status = bb_task_status(rank);
    } while (!status);
    if (status != 1 || task_dma_results[rank] != 1)
      return 4;
    for (unsigned i = 0; i < DMA_BYTES; i++)
      if (task_dma_outputs[rank][i] != (uint8_t)((rank + 1) * 17 + i * 3))
        return 5;
  }
  bb_mvover(0, 0, 0, 1, 0, 0, DMA_ROWS);
  moved = (struct dma_argument){0, (uintptr_t)task_move_output,
                                (uintptr_t)&task_move_result, 1};
  if (submit_task(1, task_move_store_kernel, &moved,
                  stacks[1] + sizeof(stacks[1]), tls[1]))
    return 6;
  uintptr_t status;
  do {
    status = bb_task_status(1);
  } while (!status);
  if (status != 1 || task_move_result != 1)
    return 7;
  for (unsigned i = 0; i < DMA_BYTES; i++)
    if (task_move_output[i] != (uint8_t)(17 + i * 3))
      return 8;
  return 0;
}
