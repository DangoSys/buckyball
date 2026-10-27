#define _GNU_SOURCE
#include "ant_shared.h"
#include <thread.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#endif
#ifdef __linux__
static void *worker(void *argument) {
  uintptr_t core = (uintptr_t)argument;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET((core + 1), &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    abort();
  shared_exercise(core);
  return NULL;
}
#endif
int main(void) {
#ifdef __linux__
  pthread_t threads[POLY_THREAD_CORES];
  for (uintptr_t core = 0; core < POLY_THREAD_CORES; ++core)
    if (pthread_create(&threads[core], NULL, worker, (void *)core))
      abort();
  for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
    if (pthread_join(threads[core], NULL))
      abort();
  printf("Ant shared/cancel PASS cores=%u\n", (unsigned)POLY_THREAD_CORES);
#else
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart != WORKLOAD_REPORT_HART)
    shared_exercise(hart - 1);
  else
    for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
      while (!__atomic_load_n(&shared_done[core], __ATOMIC_ACQUIRE)) {
      }
#endif
  return 0;
}
