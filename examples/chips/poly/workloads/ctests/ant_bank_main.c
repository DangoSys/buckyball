#define _GNU_SOURCE
#include "ant_bank.h"
#include <ant.h>
#include <params.h>
#include <topology.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#endif
#ifdef __linux__
static void *worker(void *argument) {
  uintptr_t tile = (uintptr_t)argument;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(BB_MAIN_CORES + tile, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    abort();
  bank_exercise(tile);
  return NULL;
}
#endif
int main(void) {
#ifdef __linux__
  if (mlockall(MCL_CURRENT | MCL_FUTURE))
    abort();
  pthread_t threads[BB_COMPUTE_TILES];
  for (uintptr_t tile = 0; tile < BB_COMPUTE_TILES; ++tile)
    if (pthread_create(&threads[tile], NULL, worker, (void *)tile))
      abort();
  for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
    if (pthread_join(threads[tile], NULL))
      abort();
  printf("Ant bank boundary PASS tiles=%u private=%u shared=%u\n",
         (unsigned)BB_COMPUTE_TILES, BANK_LINES, BB_SHARED_BANK_LINES);
#else
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart >= BB_MAIN_CORES)
    bank_exercise(hart - BB_MAIN_CORES);
  else
    for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
      while (!__atomic_load_n(&bank_done[tile], __ATOMIC_ACQUIRE)) {
      }
#endif
  return 0;
}
