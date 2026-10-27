#define _GNU_SOURCE
#include "ant_bank.h"
#include <thread.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#endif
#ifdef __linux__
static void *worker(void *argument) {
  uintptr_t core = (uintptr_t)argument;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET((core + 1), &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    abort();
  bank_exercise(core);
  return NULL;
}
#endif
int main(void) {
#ifdef __linux__
  if (mlockall(MCL_CURRENT | MCL_FUTURE))
    abort();
  pthread_t threads[POLY_THREAD_CORES];
  for (uintptr_t core = 0; core < POLY_THREAD_CORES; ++core)
    if (pthread_create(&threads[core], NULL, worker, (void *)core))
      abort();
  for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
    if (pthread_join(threads[core], NULL))
      abort();
  printf("Ant bank boundary PASS cores=%u private=%u shared=%u\n",
         (unsigned)POLY_THREAD_CORES, THREAD_MEM_BANK_ENTRIES,
         CORE_SHARED_MEM_BANK_ENTRIES);
#else
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == WORKLOAD_REPORT_HART) {
    for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
      while (!__atomic_load_n(&bank_done[core], __ATOMIC_ACQUIRE)) {
      }
  } else {
    bank_exercise(hart - 1);
  }
#endif
  return 0;
}
