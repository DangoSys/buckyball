#include "copy.h"
#include <engine.h>

static uint64_t input[POLY_THREAD_CORES][16] __attribute__((aligned(64)));
static uint64_t intermediate[POLY_THREAD_CORES][8][16]
    __attribute__((aligned(64)));
static uint64_t output[POLY_THREAD_CORES][16] __attribute__((aligned(64)));
static unsigned done[POLY_THREAD_CORES];
static struct engine_job jobs[POLY_THREAD_CORES][5];

int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == WORKLOAD_REPORT_HART) {
    for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
      while (!__atomic_load_n(&done[core], __ATOMIC_ACQUIRE)) {
      }
    return 0;
  }
  unsigned core = hart - 1;
  struct engine engine;
  struct engine_slot slots[5];
  engine_init(&engine, slots, 5);
  if (engine.thread_count != 5)
    return 1;
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    for (unsigned word = 0; word < 16; ++word) {
      input[core][word] = ((uint64_t)(hart + iteration) << 32) | (word + 1);
      output[core][word] = 0;
      for (unsigned buffer = 0; buffer < 8; ++buffer)
        intermediate[core][buffer][word] = UINT64_C(0xa5a5a5a5a5a5a5a5);
    }
    struct copy_task args[5];
    struct engine_result results[5];
    struct engine_transfer edges[4];
    for (unsigned thread = 0; thread < 5; ++thread) {
      args[thread] = (struct copy_task){
          (uintptr_t)(thread ? intermediate[core][2 * thread - 1]
                             : input[core]),
          (uintptr_t)(thread == 4 ? output[core]
                                  : intermediate[core][2 * thread]),
          8};
      jobs[core][thread].thread = thread;
      jobs[core][thread].task.id = thread + 1;
      jobs[core][thread].task.signature = slots[thread].signature;
      jobs[core][thread].opaque = &args[thread];
      if (thread)
        edges[thread - 1] = (struct engine_transfer){thread - 1,
                                                     thread,
                                                     args[thread - 1].output,
                                                     args[thread].input,
                                                     sizeof(input[core]),
                                                     0};
    }
    if (engine_run(&engine, jobs[core], 5, edges, 4, results))
      return 2;
    for (unsigned thread = 0; thread < 5; ++thread)
      if (results[thread].state != ENGINE_SUCCEEDED || results[thread].value)
        return 3;
    for (unsigned word = 0; word < 16; ++word) {
      if (output[core][word] != input[core][word])
        return 4;
      for (unsigned buffer = 0; buffer < 8; ++buffer)
        if (intermediate[core][buffer][word] != UINT64_C(0xa5a5a5a5a5a5a5a5))
          return 5;
    }
  }
  __atomic_store_n(&done[core], 1, __ATOMIC_RELEASE);
  return 0;
}
