#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <math.h>
#include <stdio.h>

#define N 16
#define SCALE 0x3E800000U /* 0.25f */

static const int8_t input[N] __attribute__((aligned(64))) = {
    7, 0, 1, 0, 14, -1, 1, -3, 15, -3, -3, -3, 16, -7, -3, -2,
};
static const float expected[N] = {
    1.75f, 0.00f,  0.25f,  0.00f,  3.50f, -0.25f, 0.25f,  -0.75f,
    3.75f, -0.75f, -0.75f, -0.75f, 4.00f, -1.75f, -0.75f, -0.50f,
};
static float actual[N] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mvin((uintptr_t)input, 0, 1, 1);
  bb_int2fp(0, 1, 1, SCALE);
  bb_mvout((uintptr_t)actual, 1, 1, 1);
  bb_fence();

  int passed = 1;
  for (int i = 0; i < N; ++i) {
    if (fabsf(actual[i] - expected[i]) > 1e-6f) {
      printf("FAIL i=%d exp=%f got=%f\n", i, expected[i], actual[i]);
      passed = 0;
    }
  }
  printf("quant_int8_to_fp32 %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
