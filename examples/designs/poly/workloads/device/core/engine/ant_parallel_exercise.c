#include "ant_parallel.h"
#include <thread.h>
extern const unsigned char ant_image[], ant_image_end[];
static uint64_t input[POLY_THREAD_CORES][2][512]
    __attribute__((aligned(4096), section(".noinit")));
static uint64_t output[POLY_THREAD_CORES][2][512]
    __attribute__((aligned(4096), section(".noinit")));
unsigned parallel_done[POLY_THREAD_CORES];
static unsigned reached[3] __attribute__((aligned(64)));
static void barrier(unsigned phase) {
  __atomic_fetch_add(&reached[phase], 1, __ATOMIC_ACQ_REL);
  while (__atomic_load_n(&reached[phase], __ATOMIC_ACQUIRE) !=
         POLY_THREAD_CORES) {
  }
}
void parallel_exercise(unsigned core) {
  const uint32_t contexts[] = {thread_find(THREAD_SIGNATURE, 0),
                               thread_find(THREAD_SIGNATURE, 1)};
  size_t bytes = ant_image_end - ant_image;
  uint64_t shared = thread_query(contexts[0], THREAD_FIELD_TSS_BASE), zero = 0;
  struct thread_task tasks[2];
  for (unsigned context = 0; context < 2; ++context) {
    for (unsigned word = 0; word < 2; ++word) {
      input[core][context][word] =
          ((uint64_t)(core + 1) << 32) | (context * 2 + word + 1);
      output[core][context][word] = ~0ULL;
    }
    uint64_t base = thread_query(contexts[context], THREAD_FIELD_TLS_BASE);
    if (bytes > thread_query(contexts[context], THREAD_FIELD_CODE_BYTES) ||
        thread_query(contexts[context], THREAD_FIELD_SIGNATURE) !=
            THREAD_SIGNATURE)
      __builtin_trap();
    struct ant_parallel_args args = {shared, context,
                                     (uintptr_t)input[core][context],
                                     (uintptr_t)output[core][context]};
    thread_tss_write(0, ant_image, bytes);
    thread_load(contexts[context], THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
    thread_tss_write(0, &args, sizeof(args));
    thread_load(contexts[context], THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
    tasks[context] = (struct thread_task){
        context + 1,
        0,
        bytes,
        base,
        base + thread_query(contexts[context], THREAD_FIELD_TLS_BYTES),
        THREAD_SIGNATURE};
  }
  thread_tss_write(0, &zero, sizeof(zero));
  barrier(0);
  unsigned peer = (core + 1) % POLY_THREAD_CORES;
  for (unsigned context = 0; context < 2; ++context)
    for (unsigned word = 0; word < 2; ++word)
      if (*(volatile uint64_t *)&output[peer][context][word] != ~0ULL)
        __builtin_trap();
  barrier(1);
  thread_acquire();
  thread_start(contexts[0], &tasks[0]);
  struct ant_parallel_args pending = {shared, 1, (uintptr_t)input[core][1],
                                      (uintptr_t)output[core][1]};
  thread_tss_write(64, &pending, sizeof(pending));
  thread_load(contexts[1], THREAD_TLS, 0, 64, sizeof(pending));
  thread_start(contexts[1], &tasks[1]);
  for (unsigned context = 0; context < 2; ++context) {
    int cancelled;
    if (thread_wait(contexts[context], tasks[context].id, &cancelled) ||
        cancelled)
      __builtin_trap();
    for (unsigned word = 0; word < 2; ++word)
      if (output[core][context][word] != input[core][context][word])
        __builtin_trap();
  }
  thread_release();
  barrier(2);
  for (unsigned context = 0; context < 2; ++context)
    for (unsigned word = 0; word < 2; ++word)
      if (*(volatile uint64_t *)&output[peer][context][word] !=
          input[peer][context][word])
        __builtin_trap();
  __atomic_store_n(&parallel_done[core], 1, __ATOMIC_RELEASE);
}
