#define _GNU_SOURCE
#include "ant_mxmm.h"
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
extern const unsigned char ant_image[], ant_image_end[];
struct matrices {
  float a[BB_COMPUTE_TILES][ANT_M * ANT_K];
  float b[BB_COMPUTE_TILES][ANT_N * ANT_K];
  float out[BB_COMPUTE_TILES][ANT_M * ANT_N];
};
static constexpr matrices prepare() {
  matrices data{};
  for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile) {
    for (unsigned row = 0; row < ANT_M; ++row)
      for (unsigned k = 0; k < ANT_K; ++k)
        data.a[tile][row * ANT_K + k] = row + tile + 1;
    for (unsigned col = 0; col < ANT_N; ++col)
      for (unsigned k = 0; k < ANT_K; ++k)
        data.b[tile][col * ANT_K + k] = col + 1;
    for (unsigned i = 0; i < ANT_M * ANT_N; ++i)
      data.out[tile][i] = -1;
  }
  return data;
}
alignas(64) static matrices data = prepare();
static unsigned done[BB_COMPUTE_TILES];
static void exercise(unsigned tile) {
  size_t bytes = ant_image_end - ant_image;
  if (bytes > ant_query(0, ANT_CODE_BYTES) ||
      ant_query(0, ANT_SIGNATURE) != CORE_SIGNATURE)
    __builtin_trap();
  struct ant_mxmm_args args = {(uintptr_t)data.a[tile], (uintptr_t)data.b[tile],
                               (uintptr_t)data.out[tile]};
  ant_write(0, ANT_CODE, 0, ant_image, bytes);
  ant_write(0, ANT_TLS, 0, &args, sizeof(args));
  uint64_t base = ant_query(0, ANT_TLS_BASE),
           capacity = ant_query(0, ANT_TLS_BYTES);
  struct ant_task task = {tile + 1,      0, bytes, base, base + capacity,
                          CORE_SIGNATURE};
  ant_acquire();
  ant_start(0, &task);
  int cancelled;
  if (ant_wait(0, task.id, &cancelled) || cancelled)
    __builtin_trap();
  ant_release();
  for (unsigned row = 0; row < ANT_M; ++row)
    for (unsigned col = 0; col < ANT_N; ++col)
      if (data.out[tile][row * ANT_N + col] !=
          (float)(ANT_K * (row + tile + 1) * (col + 1)))
        __builtin_trap();
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
  if (mlockall(MCL_CURRENT | MCL_FUTURE))
    abort();
  pthread_t threads[BB_COMPUTE_TILES];
  for (uintptr_t tile = 0; tile < BB_COMPUTE_TILES; ++tile)
    if (pthread_create(&threads[tile], NULL, worker, (void *)tile))
      abort();
  for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
    if (pthread_join(threads[tile], NULL))
      abort();
  printf("Ant Mxmm PASS tiles=%u\n", (unsigned)BB_COMPUTE_TILES);
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
