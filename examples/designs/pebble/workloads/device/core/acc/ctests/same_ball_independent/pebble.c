#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/smatmul.h>
#include <isa/transpose.h>
#include <multithread.h>
#include <stdint.h>
#include <stdio.h>

enum { M = 8, N = 16, K = 64, LANES = THREAD_MEM_BANK_WIDTH / 8 };
static int8_t a0[BB_NUM_CORES][M * K] __attribute__((aligned(64)));
static int8_t b0[BB_NUM_CORES][K * N] __attribute__((aligned(64)));
static int8_t a1[BB_NUM_CORES][M * K] __attribute__((aligned(64)));
static int8_t b1[BB_NUM_CORES][K * N] __attribute__((aligned(64)));
static int32_t bias[BB_NUM_CORES][N] __attribute__((aligned(64)));
static int32_t out0[BB_NUM_CORES][M * N] __attribute__((aligned(64)));
static int32_t out1[BB_NUM_CORES][M * N] __attribute__((aligned(64)));
static volatile int32_t warmup[BB_NUM_CORES][N] __attribute__((aligned(64)));
extern "C" int pebble_pipeline(uint32_t instance) {
  int c = instance;
  for (int i = 0; i < M * K; ++i) {
    a0[c][i] = 1;
    a1[c][i] = 3;
  }
  for (int i = 0; i < K * N; ++i) {
    b0[c][i] = 2;
    b1[c][i] = 4;
  }
  for (int bank = 0; bank < 7; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)a0[c], 0, M * K / LANES, 1);
  bb_mvin((uintptr_t)b0[c], 1, K * N / LANES, 1);
  bb_mvin((uintptr_t)a1[c], 3, M * K / LANES, 1);
  bb_mvin((uintptr_t)b1[c], 4, K * N / LANES, 1);
  bb_mvin((uintptr_t)bias[c], 6, N * 4 / LANES, 1);
  bb_smatmul_bias(6, 0);
  bb_smatmul_os(0, 1, 2, 1, N, N, 1, 1, 0);
  bb_mvout((uintptr_t)warmup[c], 2, N * 4 / LANES, 1);
  if (warmup[c][0] != 2 * N)
    return 1;
  uint64_t start, end;
  asm volatile("rdcycle %0" : "=r"(start));
  bb_smatmul_os(0, 1, 2, M, N, K, 1, 1, 0);
  bb_smatmul_os(3, 4, 5, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)out0[c], 2, M * N * 4 / LANES, 1);
  bb_mvout((uintptr_t)out1[c], 5, M * N * 4 / LANES, 1);
  for (int i = 0; i < M * N; ++i)
    if (out0[c][i] != 2 * K || out1[c][i] != 12 * K)
      return 1;
  asm volatile("rdcycle %0" : "=r"(end));
  for (int bank = 0; bank < 7; ++bank)
    bb_mem_release(bank);
  printf("same_ball_independent PASS cycles=%llu\n",
         (unsigned long long)(end - start));
  return 0;
}
