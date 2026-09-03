#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/int2fp.h>
#include <math.h>
#include <params.h>
#include <stdio.h>
#include <stdlib.h>

#define LANES (BANK_WIDTH / 8)
#define NELEM (BANK_LINES * LANES)
#define DA 1.0f
#define DW 1.0f
#define SEED 0xCA

static int32_t input_i32[NELEM] __attribute__((aligned(64)));
static float actual[NELEM] __attribute__((aligned(64)));
static float expected[NELEM];
static const float da_scale = DA;
static const float dw_scale = DW;

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  srand(SEED);
  for (int i = 0; i < NELEM; ++i) {
    int8_t v = (int8_t)((rand() % 256) - 128);
    input_i32[i] = v;
    expected[i] = (float)v;
    actual[i] = 0.0f;
  }
  bb_mvin_mmio((uintptr_t)(&da_scale), 0, 1, 4);
  bb_mvin_mmio((uintptr_t)(&dw_scale), BANK_WIDTH / 8, 1, 4);
  bb_mem_alloc(0, BANK_LINES, 1);
  bb_mem_alloc(1, BANK_LINES, 1);
  bb_mem_alloc(2, BANK_LINES, 1);
  bb_mvin((uintptr_t)input_i32, 0, BANK_LINES, 1);
  bb_mvin((uintptr_t)&dw_scale, 1, 4, 1);
  bb_int32_to_fp32(0, 1, 2, BANK_LINES, 0);
  bb_mvout((uintptr_t)actual, 2, BANK_LINES, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);

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
