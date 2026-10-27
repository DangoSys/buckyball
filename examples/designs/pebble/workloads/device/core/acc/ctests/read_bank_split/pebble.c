#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/smatmul.h>
#include <isa/transpose.h>
#include <multithread.h>
#include <stdint.h>
#include <stdio.h>

enum { M = 8, N = 16, K = 16, LANES = THREAD_MEM_BANK_WIDTH / 8 };
static int8_t input[BB_NUM_CORES][K * N] __attribute__((aligned(64)));
static int8_t weight[BB_NUM_CORES][K * N] __attribute__((aligned(64)));
static int32_t bias[BB_NUM_CORES][N] __attribute__((aligned(64)));
static int8_t transposed[BB_NUM_CORES][K * N] __attribute__((aligned(64)));
static int32_t product[BB_NUM_CORES][M * N] __attribute__((aligned(64)));
static volatile int32_t warmup[BB_NUM_CORES][N] __attribute__((aligned(64)));
extern "C" int pebble_pipeline(uint32_t instance) {
  int c = instance;
  for (int i = 0; i < K * N; ++i) {
    input[c][i] = 1;
    weight[c][i] = 2;
  }
  for (int bank = 0; bank < 6; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)input[c], 0, K * N / LANES, 1);
  bb_mvin((uintptr_t)input[c], 4, K * N / LANES, 1);
  bb_mvin((uintptr_t)weight[c], 2, K * N / LANES, 1);
  bb_mvin((uintptr_t)bias[c], 5, N * 4 / LANES, 1);
  bb_smatmul_bias(5, 0);
  bb_smatmul_os(0, 2, 3, 1, N, N, 1, 1, 0);
  bb_mvout((uintptr_t)warmup[c], 3, N * 4 / LANES, 1);
  if (warmup[c][0] != 2 * N)
    return 1;
  uint64_t start, end;
  asm volatile("rdcycle %0" : "=r"(start));
  bb_transpose(0, 1, K * N / LANES, 8);
  bb_smatmul_os(4, 2, 3, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)transposed[c], 1, K * N / LANES, 1);
  bb_mvout((uintptr_t)product[c], 3, M * N * 4 / LANES, 1);
  for (int i = 0; i < K * N; ++i)
    if (transposed[c][i] != 1)
      return 1;
  for (int i = 0; i < M * N; ++i)
    if (product[c][i] != 2 * K)
      return 1;
  asm volatile("rdcycle %0" : "=r"(end));
  for (int bank = 0; bank < 6; ++bank)
    bb_mem_release(bank);
  printf("read_bank_split PASS cycles=%llu\n",
         (unsigned long long)(end - start));
  return 0;
}
