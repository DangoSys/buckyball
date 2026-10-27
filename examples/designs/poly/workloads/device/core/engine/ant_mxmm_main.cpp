#define _GNU_SOURCE
#include "ant_mxmm.h"
#include <thread.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#endif
extern const unsigned char ant_image[], ant_image_end[];
struct matrices {
  float a[POLY_THREAD_CORES][ANT_M * ANT_K];
  float b[POLY_THREAD_CORES][ANT_N * ANT_K];
  float out[POLY_THREAD_CORES][ANT_M * ANT_N];
};
static constexpr matrices prepare() {
  matrices data{};
  for (unsigned core = 0; core < POLY_THREAD_CORES; ++core) {
    for (unsigned row = 0; row < ANT_M; ++row)
      for (unsigned k = 0; k < ANT_K; ++k)
        data.a[core][row * ANT_K + k] = row + core + 1;
    for (unsigned col = 0; col < ANT_N; ++col)
      for (unsigned k = 0; k < ANT_K; ++k)
        data.b[core][col * ANT_K + k] = col + 1;
    for (unsigned i = 0; i < ANT_M * ANT_N; ++i)
      data.out[core][i] = -1;
  }
  return data;
}
alignas(64) static matrices data = prepare();
static unsigned done[POLY_THREAD_CORES];
static void exercise(unsigned core) {
  const uint32_t context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = ant_image_end - ant_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  struct ant_mxmm_args args = {(uintptr_t)data.a[core], (uintptr_t)data.b[core],
                               (uintptr_t)data.out[core]};
  thread_tss_write(0, ant_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  thread_tss_write(0, &args, sizeof(args));
  thread_load(context, THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE),
           capacity = thread_query(context, THREAD_FIELD_TLS_BYTES);
  struct thread_task task = {core + 1,        0, bytes, base, base + capacity,
                             THREAD_SIGNATURE};
  thread_acquire();
  thread_start(context, &task);
  int cancelled;
  if (thread_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
  for (unsigned row = 0; row < ANT_M; ++row)
    for (unsigned col = 0; col < ANT_N; ++col)
      if (data.out[core][row * ANT_N + col] !=
          (float)(ANT_K * (row + core + 1) * (col + 1)))
        __builtin_trap();
  __atomic_store_n(&done[core], 1, __ATOMIC_RELEASE);
}
#ifdef __linux__
static void *worker(void *argument) {
  uintptr_t core = (uintptr_t)argument;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET((core + 1), &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
    abort();
  exercise(core);
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
  printf("Ant Mxmm PASS cores=%u\n", (unsigned)POLY_THREAD_CORES);
#else
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == WORKLOAD_REPORT_HART) {
    for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
      while (!__atomic_load_n(&done[core], __ATOMIC_ACQUIRE)) {
      }
  } else {
    exercise(hart - 1);
  }
#endif
  return 0;
}
