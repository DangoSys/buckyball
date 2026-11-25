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
  uintptr_t hart = (uintptr_t)argument;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET((hart + 1), &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    abort();
  shared_exercise(hart);
  return NULL;
}
#endif
int main(void) {
#ifdef __linux__
  pthread_t threads[POLY_THREAD_HARTS];
  for (uintptr_t hart = 0; hart < POLY_THREAD_HARTS; ++hart)
    if (pthread_create(&threads[hart], NULL, worker, (void *)hart))
      abort();
  for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
    if (pthread_join(threads[hart], NULL))
      abort();
  printf("Ant shared/cancel PASS harts=%u\n", (unsigned)POLY_THREAD_HARTS);
#else
  uintptr_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid != WORKLOAD_REPORT_HART)
    shared_exercise(hartid - 1);
  else
    for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
      while (!__atomic_load_n(&shared_done[hart], __ATOMIC_ACQUIRE)) {
      }
#endif
  return 0;
}
