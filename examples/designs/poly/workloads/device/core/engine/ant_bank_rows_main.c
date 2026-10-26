#define _GNU_SOURCE
#include "ant_bank_rows.h"
#include <instances.h>
#include <params.h>
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
  bank_rows_exercise(core);
  return NULL;
}
#endif
int main(void) {
#ifdef __linux__
  if (mlockall(MCL_CURRENT | MCL_FUTURE))
    abort();
  pthread_t threads[BB_THREAD_CORE_COUNT];
  for (uintptr_t core = 0; core < BB_THREAD_CORE_COUNT; ++core)
    if (pthread_create(&threads[core], NULL, worker, (void *)core))
      abort();
  for (unsigned core = 0; core < BB_THREAD_CORE_COUNT; ++core)
    if (pthread_join(threads[core], NULL))
      abort();
  printf("Ant bank boundary PASS cores=%u private=%u shared=%u\n",
         (unsigned)BB_THREAD_CORE_COUNT, BANK_LINES, BB_SHARED_BANK_LINES);
#else
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == BB_REPORT_HART) {
    for (unsigned core = 0; core < BB_THREAD_CORE_COUNT; ++core)
      while (!__atomic_load_n(&bank_rows_done[core], __ATOMIC_ACQUIRE)) {
      }
  } else {
    bank_rows_exercise(hart - 1);
  }
#endif
  return 0;
}
