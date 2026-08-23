#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/relu.h>
#include <stdio.h>

#define ROWS 4
#define COLS 16

static elem_t input[ROWS * COLS] __attribute__((aligned(64))) = {
    -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,
    -17, -16, -15, -14, -13, -12, -11, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    -27, -26, -25, -24, -23, -22, -21, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    -37, -36, -35, -34, -33, -32, -31, 30, 31, 32, 33, 34, 35, 36, 37, 38,
};

static elem_t expected[ROWS * COLS] __attribute__((aligned(64)));
static elem_t output[ROWS * COLS] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  for (int i = 0; i < ROWS * COLS; i++) {
    expected[i] = input[i] < 0 ? 0 : input[i];
  }

  const uint32_t src = 0;
  const uint32_t dst = 1;
  bb_mem_alloc(src, 1, 1);
  bb_mem_alloc(dst, 1, 1);
  bb_mvin((uintptr_t)input, src, ROWS, 1);
  bb_relu(src, dst, ROWS);
  bb_mvout((uintptr_t)output, dst, ROWS, 1);
  bb_fence();
  bb_mem_release(src);
  bb_mem_release(dst);

  int passed = compare_i8_matrices(output, expected, ROWS, COLS);
  printf("ReLU test %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
