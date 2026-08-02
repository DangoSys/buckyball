#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <bbhw/mem/params.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define LANES 16
#define NELEM (BANK_LINES * LANES)
#define SCALE 0x3F800000U /* 1.0f */
#define SEED 0xCA

static int8_t input_i8[NELEM] __attribute__((aligned(64)));
static float actual[NELEM] __attribute__((aligned(64)));
static float expected[NELEM];

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  srand(SEED);
  for (int i = 0; i < NELEM; ++i) {
    int8_t v = (int8_t)((rand() % 256) - 128);
    input_i8[i] = v;
    expected[i] = (float)v;
    actual[i] = 0.0f;
  }
  bb_mem_alloc(0, BANK_LINES, 1);
  bb_mem_alloc(1, BANK_LINES, 4);
  bb_mvin((uintptr_t)input_i8, 0, BANK_LINES, 1);
  bb_int2fp(0, 1, BANK_LINES, SCALE);
  bb_mvout((uintptr_t)actual, 1, BANK_LINES, 1);
  bb_fence();

  int passed = 1;
  for (int i = 0; i < NELEM; ++i) {
    if (fabsf(actual[i] - expected[i]) > 1e-6f) {
      printf("FAIL i=%d exp=%f got=%f\n", i, expected[i], actual[i]);
      passed = 0;
      break;
    }
  }
  printf("quant_int8_to_fp32_bank %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
