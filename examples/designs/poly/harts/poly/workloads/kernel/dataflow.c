#include "copy.h"
#include <kernel.h>

static uint64_t input[POLY_THREAD_HARTS][16] __attribute__((aligned(64)));
static uint64_t intermediate[POLY_THREAD_HARTS][8][16]
    __attribute__((aligned(64)));
static uint64_t output[POLY_THREAD_HARTS][16] __attribute__((aligned(64)));
static unsigned done[POLY_THREAD_HARTS];
static struct kernel_job jobs[POLY_THREAD_HARTS][5];

int main(void) {
  uintptr_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid == WORKLOAD_REPORT_HART) {
    for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
      while (!__atomic_load_n(&done[hart], __ATOMIC_ACQUIRE)) {
      }
    return 0;
  }
  unsigned hart = hartid - 1;
  struct kernel kernel;
  struct kernel_slot slots[5];
  kernel_init(&kernel, slots, 5);
  if (kernel.thread_count != 5)
    return 1;
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    for (unsigned word = 0; word < 16; ++word) {
      input[hart][word] = ((uint64_t)(hartid + iteration) << 32) | (word + 1);
      output[hart][word] = 0;
      for (unsigned buffer = 0; buffer < 8; ++buffer)
        intermediate[hart][buffer][word] = UINT64_C(0xa5a5a5a5a5a5a5a5);
    }
    struct copy_task args[5];
    struct kernel_result results[5];
    struct kernel_transfer edges[4];
    for (unsigned thread = 0; thread < 5; ++thread) {
      args[thread] = (struct copy_task){
          (uintptr_t)(thread ? intermediate[hart][2 * thread - 1]
                             : input[hart]),
          (uintptr_t)(thread == 4 ? output[hart]
                                  : intermediate[hart][2 * thread]),
          8};
      jobs[hart][thread].thread = thread;
      jobs[hart][thread].task.id = thread + 1;
      jobs[hart][thread].task.signature = slots[thread].signature;
      jobs[hart][thread].opaque = &args[thread];
      if (thread)
        edges[thread - 1] = (struct kernel_transfer){thread - 1,
                                                     thread,
                                                     args[thread - 1].output,
                                                     args[thread].input,
                                                     sizeof(input[hart]),
                                                     0};
    }
    if (kernel_run(&kernel, jobs[hart], 5, edges, 4, results))
      return 2;
    for (unsigned thread = 0; thread < 5; ++thread)
      if (results[thread].state != KERNEL_SUCCEEDED || results[thread].value)
        return 3;
    for (unsigned word = 0; word < 16; ++word) {
      if (output[hart][word] != input[hart][word])
        return 4;
      for (unsigned buffer = 0; buffer < 8; ++buffer)
        if (intermediate[hart][buffer][word] != UINT64_C(0xa5a5a5a5a5a5a5a5))
          return 5;
    }
  }
  __atomic_store_n(&done[hart], 1, __ATOMIC_RELEASE);
  return 0;
}
