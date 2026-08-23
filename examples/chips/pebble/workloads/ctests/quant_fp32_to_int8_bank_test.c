#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <bbhw/mem/params.h>
#include <isa/fp2int.h>
#include <stdio.h>
#include <stdlib.h>

#define LANES 16
#define NELEM (BANK_LINES * LANES)
#define SEED 0xC5

static float input_fp32[NELEM] __attribute__((aligned(64)));
static int8_t actual[NELEM] __attribute__((aligned(64)));
static int8_t expected[NELEM];

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  srand(SEED);
  for (int i = 0; i < NELEM; ++i) {
    int v = (rand() % 201) - 100; /* integer-valued float in [-100,100] */
    input_fp32[i] = (float)v;
    actual[i] = (int8_t)0x5a;
  }
  int max_abs = 0;
  for (int i = 0; i < NELEM; ++i) {
    int value = (int)input_fp32[i];
    int magnitude = value < 0 ? -value : value;
    if (magnitude > max_abs)
      max_abs = magnitude;
  }
  for (int i = 0; i < NELEM; ++i) {
    int value = (int)input_fp32[i];
    int magnitude = value < 0 ? -value : value;
    int numerator = magnitude * 127;
    int quantized = numerator / max_abs;
    int remainder = numerator % max_abs;
    if (2 * remainder > max_abs ||
        (2 * remainder == max_abs && (quantized & 1)))
      ++quantized;
    expected[i] = (int8_t)(value < 0 ? -quantized : quantized);
  }
  bb_mem_alloc(0, BANK_LINES, 4);
  bb_mem_alloc(1, BANK_LINES, 1);
  bb_mvin((uintptr_t)input_fp32, 0, BANK_LINES, 1);
  bb_fp2int(0, 1, BANK_LINES, 0);
  bb_mvout((uintptr_t)actual, 1, BANK_LINES, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);

  int passed = 1;
  for (int i = 0; i < NELEM; ++i) {
    if (actual[i] != expected[i]) {
      printf("FAIL i=%d exp=%d got=%d\n", i, (int)expected[i], (int)actual[i]);
      passed = 0;
      break;
    }
  }
  printf("quant_fp32_to_int8_bank %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
