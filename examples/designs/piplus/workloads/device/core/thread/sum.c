#include "sum.h"

static uint64_t mcycle(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mcycle" : "=r"(value));
  return value;
}

/* Runs on the core's Ant thread: a weighted sum of the first words its host
 * passed, into the task; returns the thread's own cycles in it. */
uint64_t entry(struct sum_task *task) {
  uint64_t start = mcycle();
  uint64_t total = 0;
  for (unsigned i = 0; i < task->count; ++i)
    total += task->words[i] * (i + 1);
  task->total = total;
  return mcycle() - start;
}
