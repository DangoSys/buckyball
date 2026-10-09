#include "ant_parallel.h"
#include <ant.h>
#include <params.h>
#include <topology.h>
extern const unsigned char ant_image[], ant_image_end[];
static uint64_t input[BB_COMPUTE_TILES][2][512]
    __attribute__((aligned(4096), section(".noinit")));
static uint64_t output[BB_COMPUTE_TILES][2][512]
    __attribute__((aligned(4096), section(".noinit")));
unsigned parallel_done[BB_COMPUTE_TILES];
static unsigned reached[3] __attribute__((aligned(64)));
static void barrier(unsigned phase) {
  __atomic_fetch_add(&reached[phase], 1, __ATOMIC_ACQ_REL);
  while (__atomic_load_n(&reached[phase], __ATOMIC_ACQUIRE) !=
         BB_COMPUTE_TILES) {
  }
}
void parallel_exercise(unsigned tile) {
  size_t bytes = ant_image_end - ant_image;
  uint64_t shared = ant_query(0, ANT_TSS_BASE), zero = 0;
  struct ant_task tasks[2];
  for (unsigned context = 0; context < 2; ++context) {
    for (unsigned word = 0; word < 2; ++word) {
      input[tile][context][word] =
          ((uint64_t)(tile + 1) << 32) | (context * 2 + word + 1);
      output[tile][context][word] = ~0ULL;
    }
    uint64_t base = ant_query(context, ANT_TLS_BASE);
    if (bytes > ant_query(context, ANT_CODE_BYTES) ||
        ant_query(context, ANT_SIGNATURE) != CORE_SIGNATURE)
      __builtin_trap();
    struct ant_parallel_args args = {shared, context,
                                     (uintptr_t)input[tile][context],
                                     (uintptr_t)output[tile][context]};
    ant_write(context, ANT_CODE, 0, ant_image, bytes);
    ant_write(context, ANT_TLS, 0, &args, sizeof(args));
    tasks[context] = (struct ant_task){
        context + 1,   0, bytes, base, base + ant_query(context, ANT_TLS_BYTES),
        CORE_SIGNATURE};
  }
  ant_write(0, ANT_TSS, 0, &zero, sizeof(zero));
  barrier(0);
  unsigned peer = (tile + 1) % BB_COMPUTE_TILES;
  for (unsigned context = 0; context < 2; ++context)
    for (unsigned word = 0; word < 2; ++word)
      if (*(volatile uint64_t *)&output[peer][context][word] != ~0ULL)
        __builtin_trap();
  barrier(1);
  ant_acquire();
  ant_start(0, &tasks[0]);
  ant_start(1, &tasks[1]);
  for (unsigned context = 0; context < 2; ++context) {
    int cancelled;
    if (ant_wait(context, tasks[context].id, &cancelled) || cancelled)
      __builtin_trap();
    for (unsigned word = 0; word < 2; ++word)
      if (output[tile][context][word] != input[tile][context][word])
        __builtin_trap();
  }
  ant_release();
  barrier(2);
  for (unsigned context = 0; context < 2; ++context)
    for (unsigned word = 0; word < 2; ++word)
      if (*(volatile uint64_t *)&output[peer][context][word] !=
          input[peer][context][word])
        __builtin_trap();
  __atomic_store_n(&parallel_done[tile], 1, __ATOMIC_RELEASE);
}
