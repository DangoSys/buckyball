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
  uintptr_t hart = (uintptr_t)argument;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET((hart + 1), &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    abort();
  bank_exercise(hart);
  return NULL;
}
#endif
int main(void) {
#ifdef __linux__
  if (mlockall(MCL_CURRENT | MCL_FUTURE))
    abort();
  pthread_t threads[POLY_THREAD_HARTS];
  for (uintptr_t hart = 0; hart < POLY_THREAD_HARTS; ++hart)
    if (pthread_create(&threads[hart], NULL, worker, (void *)hart))
      abort();
  for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
    if (pthread_join(threads[hart], NULL))
      abort();
  printf("Ant bank boundary PASS harts=%u private=%u\n",
         (unsigned)POLY_THREAD_HARTS, THREAD_MEM_BANK_ENTRIES);
#else
  uintptr_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid == WORKLOAD_REPORT_HART) {
    for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
      while (!__atomic_load_n(&bank_done[hart], __ATOMIC_ACQUIRE)) {
      }
  } else {
    bank_exercise(hartid - 1);
  }
#endif
  return 0;
}
