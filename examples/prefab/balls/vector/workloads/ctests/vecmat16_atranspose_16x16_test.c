#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/vecmat16.h>
#include <stdint.h>
#include <stdio.h>

enum { DIM = 16 };
static int8_t a[DIM * DIM] __attribute__((aligned(64)));
static int8_t at[DIM * DIM] __attribute__((aligned(64)));
static int8_t b[DIM * DIM] __attribute__((aligned(64)));
static int32_t actual[DIM * DIM] __attribute__((aligned(64)));
static int32_t expected[DIM * DIM] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col) {
      a[row * DIM + col] = (row * 3 + col * 5) % 11 - 5;
      b[row * DIM + col] = (row * 7 + col * 2) % 13 - 6;
      at[col * DIM + row] = a[row * DIM + col];
    }
  cpu_matmul(a, b, expected, DIM, DIM, DIM);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 4);
  bb_mvin((uintptr_t)at, 1, DIM, 1);
  bb_mvin((uintptr_t)b, 2, DIM, 1);
  // C starts at zero: actual is zero-initialized.
  bb_mvin((uintptr_t)actual, 3, DIM, 1);
  bb_vecmat16(1, 2, 3, DIM);
  bb_mvout((uintptr_t)actual, 3, DIM, 1);
  if (!compare_i32_matrices(actual, expected, DIM, DIM)) {
    printf("vecmat16_atranspose_16x16 FAIL\n");
    return 1;
  }
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  printf("vecmat16_atranspose_16x16 PASS\n");
  return 0;
}
