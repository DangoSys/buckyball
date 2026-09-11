#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <isa/int2fp.h>
#include <isa/matadd.h>
#include <isa/smatmul.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

enum {
  TILE = 16,
  IN = 4,
  K = 3,
  OUT = 2,
  WINDOWS = 4,
  PACKED_K = 16,
  RESULT_ROWS = TILE * 2
};
static int8_t images[2][IN * IN * TILE] __attribute__((aligned(64)));
static int8_t weights[2][PACKED_K * TILE] __attribute__((aligned(64)));
static float scale[TILE] __attribute__((aligned(64)));
static int32_t bias[TILE] __attribute__((aligned(64)));
static float output[TILE * TILE] __attribute__((aligned(64)));

static int8_t image_value(int branch, int row, int column) {
  return (int8_t)(branch * 7 + row * IN + column - 8);
}

static int32_t expected(int row, int column) {
  int32_t sum = 0;
  for (int branch = 0; branch < 2; ++branch)
    for (int ky = 0; ky < K; ++ky)
      for (int kx = 0; kx < K; ++kx)
        sum += image_value(branch, row / OUT + ky, row % OUT + kx) *
               weights[branch][(ky * K + kx) * TILE + column];
  return sum;
}

static void run_branch(int branch, int out_bank) {
  bb_mvin((uintptr_t)images[branch], 0, IN * IN, 1);
  bb_im2col(0, 1, IN, K, 1, 0, 0, 0, 0, 0, 0, WINDOWS);
  bb_mvin((uintptr_t)weights[branch], 0, PACKED_K, 1);
  bb_smatmul_os(1, 0, out_bank, TILE, TILE, PACKED_K, 1, 1, 0);
  bb_fence();
}

int main(void) {
  for (int branch = 0; branch < 2; ++branch) {
    for (int row = 0; row < IN; ++row)
      for (int column = 0; column < IN; ++column)
        images[branch][(row * IN + column) * TILE] =
            image_value(branch, row, column);
    for (int i = 0; i < PACKED_K * TILE; ++i)
      weights[branch][i] = (int8_t)((i / TILE * 3 + i % TILE) % 41 - 20);
  }
  for (int lane = 0; lane < TILE; ++lane)
    scale[lane] = 0.05f;

  for (int bank = 0; bank < 8; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)scale, 3, 4, 1);
  bb_mvin((uintptr_t)bias, 7, 4, 1);
  bb_smatmul_bias(7, 0);
  run_branch(0, 2);
  run_branch(1, 4);
  bb_matadd(2, 4, 6, RESULT_ROWS);
  bb_int32_to_fp32(6, 3, 5, RESULT_ROWS, 0);
  bb_mvout((uintptr_t)output, 5, RESULT_ROWS, 1);
  bb_fence();

  for (int row = 0; row < WINDOWS; ++row)
    for (int column = 0; column < TILE; ++column)
      if (fabsf(output[row * TILE + column] -
                 expected(row, column) * scale[column]) > 1e-3f) {
        printf("dual conv add mismatch row=%d column=%d\n", row, column);
        return 1;
      }
  printf("dual conv add int2fp PASS\n");
  return 0;
}
