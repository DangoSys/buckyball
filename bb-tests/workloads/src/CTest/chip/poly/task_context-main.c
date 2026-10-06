#include "task_dma.h"
// Tasks inherit the controller's satp: every worker runs its kernel through the
// controller's Sv39 table, whose 2-MiB leaf maps the image at VA 0x80000000.
static uint64_t root[512] __attribute__((aligned(4096)));
static uint64_t pages[512] __attribute__((aligned(4096)));
static uint8_t stacks[TASK_WORKERS][4096] __attribute__((aligned(16)));
static uint8_t tls[TASK_WORKERS][64] __attribute__((aligned(16)));
static volatile uint64_t results[TASK_WORKERS];
static void task_context_kernel(void *argument) {
  uint64_t sum = 0;
  for (unsigned i = 1; i <= 32; i++)
    sum += (uint64_t)i * i;
  *(volatile uint64_t *)argument = sum;
  task_finish(0);
}
int main(void) {
  core_id_t id = bb_get_core_id();
  if (id.core)
    buckyball_task_worker();
  if (id.tile != TASK_TILE || bb_task_worker_count() != TASK_WORKERS)
    return 1;
  root[2] = ((uintptr_t)pages >> 12) << 10 | 1;
  pages[0] = (UINT64_C(0x80000000) >> 12) << 10 | 0xdf;
  uintptr_t satp = (UINT64_C(8) << 60) | ((uintptr_t)root >> 12);
  asm volatile("csrw satp, %0; sfence.vma" ::"r"(satp) : "memory");
  for (unsigned worker = 0; worker < TASK_WORKERS; worker++) {
    task_tls(tls[worker]);
    if (submit_task(worker, task_context_kernel, (void *)&results[worker],
                    stacks[worker] + sizeof(stacks[worker]), tls[worker]))
      return 2;
  }
  for (unsigned worker = 0; worker < TASK_WORKERS; worker++) {
    uintptr_t status;
    do {
      status = bb_task_status(worker);
    } while (!status);
    if (status != 1 || results[worker] != 11440)
      return 3;
  }
  asm volatile("csrw satp, zero; sfence.vma" ::: "memory");
  return 0;
}
