#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define INPUT_N 36
#define ROWS 3            /* ceil(36/16) */
#define SCALE 0x40000000U /* 2.0f */

static const float input_fp32[INPUT_N] = {
    0.5f,  1.5f,  2.5f,  3.5f,  4.5f,  5.5f,  1.5f,  0.5f,  -0.5f,
    -1.5f, -2.5f, -3.5f, 2.5f,  -0.5f, -2.5f, 1.5f,  -1.5f, 3.5f,
    3.5f,  -1.5f, 1.5f,  -2.5f, 0.5f,  -1.5f, 4.5f,  -2.5f, -1.5f,
    0.5f,  2.5f,  -0.5f, 5.5f,  -3.5f, 3.5f,  -1.5f, -0.5f, 1.5f,
};

static const int8_t expected[INPUT_N] = {
    1, 3,  5, 7,  9, 11, 3, 1,  -1, -3, -5, -7, 5,  -1, -5, 3,  -3, 7,
    7, -3, 3, -5, 1, -3, 9, -5, -3, 1,  5,  -1, 11, -7, 7,  -3, -1, 3,
};

static float packed[ROWS * 16] __attribute__((aligned(64)));
static int8_t actual[ROWS * 16] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  for (int i = 0; i < ROWS * 16; ++i)
    packed[i] = 0.0f;
  for (int i = 0; i < INPUT_N; ++i)
    packed[i] = input_fp32[i];
  bb_dma_touch(actual, sizeof(actual));

  bb_mem_alloc(0, 1, 4);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)packed, 0, ROWS, 1);
  bb_fp2int(0, 1, ROWS, SCALE);
  bb_mvout((uintptr_t)actual, 1, ROWS, 1);
  bb_fence();

  int passed = 1;
  for (int i = 0; i < INPUT_N; ++i) {
    if (actual[i] != expected[i]) {
      printf("FAIL i=%d exp=%d got=%d\n", i, (int)expected[i], (int)actual[i]);
      passed = 0;
    }
  }
  printf("quant_fp32_to_int8 %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
