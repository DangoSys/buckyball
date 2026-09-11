#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
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
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  bb_mvin((uintptr_t)at, 0, DIM, 1);
  bb_mvin((uintptr_t)b, 1, DIM, 1);
  bb_vecmat16(0, 1, 2, DIM, 0);
  bb_mvout((uintptr_t)actual, 2, DIM, 1);
  bb_fence();
  if (!compare_i32_matrices(actual, expected, DIM, DIM)) {
    printf("vecmat16_atranspose_16x16 FAIL\n");
    return 1;
  }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  printf("vecmat16_atranspose_16x16 PASS\n");
  return 0;
}
