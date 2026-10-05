#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/gemmini.h>
#include <stdio.h>

enum { DIM = 16 };
static elem_t a[DIM * DIM] __attribute__((aligned(64)));
static elem_t b[DIM * DIM] __attribute__((aligned(64)));
static int32_t initial[DIM * DIM] __attribute__((aligned(64)));
static int32_t result[DIM * DIM] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col) {
      a[row * DIM + col] = (3 * row + 5 * col) % 7 - 3;
      b[row * DIM + col] = (2 * row + 3 * col) % 5 - 2;
      initial[row * DIM + col] = 7;
    }
  for (int shift = 0; shift <= 4; shift += 4) {
    bb_mem_alloc(0, 1, 1);
    bb_mem_alloc(1, 1, 1);
    bb_mem_alloc(3, 1, 4);
    bb_mvin((uintptr_t)a, 0, DIM, 1);
    bb_mvin((uintptr_t)b, 1, DIM, 1);
    bb_mvin((uintptr_t)initial, 3, DIM, 1);
    bb_gemmini_config(1, 0, 0, 0, shift);
    bb_gemmini_compute_accumulated(0, 1, 3, DIM, 0, 0, 0);
    bb_mvout((uintptr_t)result, 3, DIM, 1);
    bb_fence();
    for (int row = 0; row < DIM; ++row)
      for (int col = 0; col < DIM; ++col) {
        int expected = 7;
        for (int k = 0; k < DIM; ++k)
          expected += a[row * DIM + k] * b[k * DIM + col];
        expected = gemmini_in_shift(expected, shift);
        if (result[row * DIM + col] != expected) {
          printf("Gemmini accumulate FAILED r=%d c=%d actual=%ld expected=%d\n",
                 row, col, (long)result[row * DIM + col], expected);
          return 1;
        }
      }
    bb_mem_release(0);
    bb_mem_release(1);
    bb_mem_release(3);
  }
  printf("Gemmini accumulate PASSED\n");
  return 0;
}
