#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

/* 4x4 input, 3x3 kernel -> 2x2 windows; keeps golden small. */
#define INPUT_H 4
#define KERNEL_H 3
#define STRIDE 1
#define PADDING 0
#define NUM_WINDOWS 4
#define KERNEL_ELEMS 9
#define QUANT_ROWS 1

static const int8_t quant_input[16] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
};

/* row-major windows of 3x3 patches from 4x4 */
static const int8_t expected[NUM_WINDOWS * KERNEL_ELEMS] = {
    1, 2, 3, 5, 6,  7,  9,  10, 11, 2, 3, 4, 6,  7,  8,  10, 11, 12,
    5, 6, 7, 9, 10, 11, 13, 14, 15, 6, 7, 8, 10, 11, 12, 14, 15, 16,
};

static int8_t packed[16] __attribute__((aligned(64)));
static int8_t actual[NUM_WINDOWS * 16] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  for (int i = 0; i < 16; ++i)
    packed[i] = quant_input[i];

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)packed, 0, QUANT_ROWS, 1);
  bb_im2col(0, 1, INPUT_H, KERNEL_H, STRIDE, PADDING);
  bb_mvout((uintptr_t)actual, 1, NUM_WINDOWS, 1);
  bb_fence();

  int passed = 1;
  for (int row = 0; row < NUM_WINDOWS; ++row) {
    for (int col = 0; col < KERNEL_ELEMS; ++col) {
      int8_t got = actual[row * 16 + col];
      int8_t exp = expected[row * KERNEL_ELEMS + col];
      if (got != exp) {
        printf("FAIL row=%d col=%d exp=%d got=%d\n", row, col, (int)exp,
               (int)got);
        passed = 0;
      }
    }
  }
  printf("quant_im2col %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
