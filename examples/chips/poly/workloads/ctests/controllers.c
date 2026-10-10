#define _GNU_SOURCE
#include <stdint.h>
#include <topology.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#endif
// Controllers publish cache-line payloads and contend on one atomic counter.
enum { CONTROLLERS = BB_MAIN_CORES + BB_COMPUTE_TILES, ROUNDS = 2 };
static uint64_t payload[CONTROLLERS][8] __attribute__((aligned(64)));
static uint64_t counter __attribute__((aligned(64)));
static uint64_t turn __attribute__((aligned(64)));
static uint64_t done[CONTROLLERS] __attribute__((aligned(64)));
static int exercise(uintptr_t hart) {
  for (unsigned round = 0; round < ROUNDS; ++round) {
    uint64_t expected = round * CONTROLLERS + hart;
    while (__atomic_load_n(&turn, __ATOMIC_ACQUIRE) != expected) {
    }
    if (expected) {
      unsigned previous = (hart + CONTROLLERS - 1) % CONTROLLERS;
      if (payload[previous][7] != expected) {
#ifdef __linux__
        exit(1);
#else
        *(volatile uint32_t *)0x60000000 = 1;
        for (;;)
          asm volatile("wfi");
#endif
      }
    }
    payload[hart][7] = expected + 1;
    __atomic_store_n(&turn, expected + 1, __ATOMIC_RELEASE);
  }
  for (unsigned i = 0; i < ROUNDS; ++i)
    __atomic_fetch_add(&counter, 1, __ATOMIC_SEQ_CST);
  __atomic_store_n(&done[hart], 1, __ATOMIC_RELEASE);
  if (hart == 0) {
    for (unsigned i = 0; i < CONTROLLERS; ++i)
      while (!__atomic_load_n(&done[i], __ATOMIC_ACQUIRE)) {
      }
    return counter == CONTROLLERS * ROUNDS ? 0 : 2;
  }
  return 0;
}

#ifdef __linux__
static void *worker(void *argument) {
  uintptr_t hart = (uintptr_t)argument;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(hart, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    exit(3);
  if (exercise(hart))
    exit(2);
  return NULL;
}
#endif
int main(void) {
#ifdef __linux__
  if (sysconf(_SC_NPROCESSORS_ONLN) != CONTROLLERS)
    return 4;
  pthread_t threads[CONTROLLERS];
  for (uintptr_t hart = 0; hart < CONTROLLERS; ++hart)
    if (pthread_create(&threads[hart], NULL, worker, (void *)hart))
      return 5;
  for (unsigned hart = 0; hart < CONTROLLERS; ++hart)
    if (pthread_join(threads[hart], NULL))
      return 6;
  printf("Controllers SMP PASS cores=%u\n", (unsigned)CONTROLLERS);
  return 0;
#else
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  return exercise(hart);
#endif
}
