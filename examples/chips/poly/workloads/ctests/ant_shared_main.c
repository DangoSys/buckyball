#define _GNU_SOURCE
#include "ant_shared.h"
#include <ant.h>
#include <params.h>
#include <topology.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#endif
extern const unsigned char ant_image[], ant_image_end[];
static unsigned done[BB_COMPUTE_TILES];
static void exercise(unsigned tile) {
  size_t bytes = ant_image_end - ant_image;
  uint64_t value = 100 + tile, shared = ant_query(0, ANT_TSS_BASE),
           zero[2] = {0};
  struct ant_task tasks[2];
  for (unsigned context = 0; context < 2; ++context) {
    uint64_t base = ant_query(context, ANT_TLS_BASE);
    if (bytes > ant_query(context, ANT_CODE_BYTES) ||
        ant_query(context, ANT_SIGNATURE) != CORE_SIGNATURE)
      __builtin_trap();
    tasks[context] = (struct ant_task){
        context + 1,   0, bytes, base, base + ant_query(context, ANT_TLS_BYTES),
        CORE_SIGNATURE};
    struct ant_shared_args args = {shared, context == 0, context ? 0 : value};
    ant_write(context, ANT_CODE, 0, ant_image, bytes);
    ant_write(context, ANT_TLS, 0, &args, sizeof(args));
  }
  ant_write(0, ANT_TSS, 0, zero, sizeof(zero));
  ant_acquire();
  ant_start(1, &tasks[1]);
  ant_start(0, &tasks[0]);
  for (unsigned context = 0; context < 2; ++context) {
    int cancelled;
    if (ant_wait(context, tasks[context].id, &cancelled) != value || cancelled)
      __builtin_trap();
    if (ant_read(context, ANT_TLS, 16) != value)
      __builtin_trap();
  }
  ant_release();
  ant_write(0, ANT_TSS, 0, zero, sizeof(zero));
  ant_acquire();
  ant_start(1, &tasks[1]);
  ant_cancel(1);
  int cancelled;
  ant_wait(1, tasks[1].id, &cancelled);
  if (!cancelled)
    __builtin_trap();
  ant_release();
  uint64_t ready[2] = {value + 7, 1};
  ant_write(0, ANT_TSS, 0, ready, sizeof(ready));
  ++tasks[1].id;
  ant_acquire();
  ant_start(1, &tasks[1]);
  if (ant_wait(1, tasks[1].id, &cancelled) != value + 7 || cancelled)
    __builtin_trap();
  ant_release();
  __atomic_store_n(&done[tile], 1, __ATOMIC_RELEASE);
}
#ifdef __linux__
static void *worker(void *argument) {
  uintptr_t tile = (uintptr_t)argument;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(BB_MAIN_CORES + tile, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    abort();
  exercise(tile);
  return NULL;
}
#endif
int main(void) {
#ifdef __linux__
  pthread_t threads[BB_COMPUTE_TILES];
  for (uintptr_t tile = 0; tile < BB_COMPUTE_TILES; ++tile)
    if (pthread_create(&threads[tile], NULL, worker, (void *)tile))
      abort();
  for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
    if (pthread_join(threads[tile], NULL))
      abort();
  printf("Ant shared/cancel PASS tiles=%u\n", (unsigned)BB_COMPUTE_TILES);
#else
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart >= BB_MAIN_CORES)
    exercise(hart - BB_MAIN_CORES);
  else
    for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
      while (!__atomic_load_n(&done[tile], __ATOMIC_ACQUIRE)) {
      }
#endif
  return 0;
}
