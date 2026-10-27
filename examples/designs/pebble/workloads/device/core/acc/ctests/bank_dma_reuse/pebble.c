#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/smatmul.h>
#include <isa/transpose.h>
#include <multithread.h>
#include <stdint.h>
#include <stdio.h>

enum { M = 8, N = 16, K = 64, LANES = THREAD_MEM_BANK_WIDTH / 8 };
static int8_t input[BB_NUM_CORES][M * K] __attribute__((aligned(64)));
static int8_t first[BB_NUM_CORES][K * N] __attribute__((aligned(64)));
static int8_t second[BB_NUM_CORES][K * N] __attribute__((aligned(64)));
static int32_t bias[BB_NUM_CORES][N] __attribute__((aligned(64)));
static int32_t out0[BB_NUM_CORES][M * N] __attribute__((aligned(64)));
static int32_t out1[BB_NUM_CORES][M * N] __attribute__((aligned(64)));
static volatile int32_t warmup[BB_NUM_CORES][N] __attribute__((aligned(64)));
static volatile int8_t moved[BB_NUM_CORES][K * N] __attribute__((aligned(64)));
extern "C" int pebble_pipeline(uint32_t instance) {
  int c = instance;
  for (int i = 0; i < M * K; ++i)
    input[c][i] = 1;
  for (int i = 0; i < K * N; ++i) {
    first[c][i] = 2;
    second[c][i] = 3;
  }
  for (int bank = 0; bank < 6; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)input[c], 0, M * K / LANES, 1);
  bb_mvin((uintptr_t)first[c], 1, K * N / LANES, 1);
  bb_mvin((uintptr_t)bias[c], 2, N * 4 / LANES, 1);
  bb_smatmul_bias(2, 0);
  bb_smatmul_os(0, 1, 3, 1, N, N, 1, 1, 0);
  bb_mvout((uintptr_t)warmup[c], 3, N * 4 / LANES, 1);
  if (warmup[c][0] != 2 * N)
    return 1;
  uint64_t start, end;
  asm volatile("rdcycle %0" : "=r"(start));
  bb_smatmul_os(0, 1, 3, M, N, K, 1, 1, 0);
  bb_mvin((uintptr_t)second[c], 1, K * N / LANES, 1);
  bb_smatmul_os(0, 1, 4, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)out0[c], 3, M * N * 4 / LANES, 1);
  bb_mvout((uintptr_t)out1[c], 4, M * N * 4 / LANES, 1);
  for (int i = 0; i < M * N; ++i)
    if (out0[c][i] != 2 * K || out1[c][i] != 3 * K) {
      printf("bank_dma_reuse FAIL index=%d actual=%d,%d\n", i, out0[c][i],
             out1[c][i]);
      return 1;
    }
  asm volatile("rdcycle %0" : "=r"(end));
  bb_mem_transfer(1, 6);
  bb_mvout((uintptr_t)moved[c], 6, K * N / LANES, 1);
  for (int i = 0; i < K * N; ++i)
    if (moved[c][i] != 3)
      return 1;
  for (int bank = 0; bank < 7; ++bank)
    if (bank != 1)
      bb_mem_release(bank);
  printf("bank_dma_reuse PASS cycles=%llu\n",
         (unsigned long long)(end - start));
  return 0;
}
