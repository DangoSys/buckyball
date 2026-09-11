#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int2fp.h>
#include <isa/quant.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum {
  M = 16,
  N = 16,
  K = 16,
  F32_ROWS = M * N * 32 / BANK_WIDTH,
  CHANNEL_ROWS = N * 32 / BANK_WIDTH
};
static float input[M * K] __attribute__((aligned(64)));
static int8_t weight[K * N] __attribute__((aligned(64)));
static int32_t bias[N] __attribute__((aligned(64)));
static float scale[N] __attribute__((aligned(64)));
static float output[M * N] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < K; ++col)
      input[row * K + col] = (float)((row * 3 + col * 5) % 15 - 7);
  for (int row = 0; row < K; ++row)
    for (int col = 0; col < N; ++col)
      weight[row * N + col] = (2 * row + 3 * col) % 5 - 2;
  for (int col = 0; col < N; ++col) {
    bias[col] = col - 8;
    scale[col] = 1.0f;
  }
  for (int bank = 0; bank < 7; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)input, 0, F32_ROWS, 1);
  bb_quant_f32_to_i8(0, 1, F32_ROWS, 1.0f);
  bb_mvin((uintptr_t)weight, 2, K, 1);
  bb_mvin((uintptr_t)bias, 3, CHANNEL_ROWS, 1);
  bb_smatmul_bias(3, 0);
  bb_smatmul_os(1, 2, 4, M, N, K, 1, 1, 0);
  bb_mvin((uintptr_t)scale, 5, CHANNEL_ROWS, 1);
  bb_int32_to_fp32(4, 5, 6, F32_ROWS, 0);
  bb_mvout((uintptr_t)output, 6, F32_ROWS, 1);
  bb_fence();
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      int expected = bias[col];
      for (int k = 0; k < K; ++k)
        expected += (int)input[row * K + k] * weight[k * N + col];
      if (output[row * N + col] != (float)expected) {
        printf("quant smatmul int2fp mismatch row=%d col=%d\n", row, col);
        return 1;
      }
    }
  printf("quant smatmul int2fp PASS\n");
  return 0;
}
