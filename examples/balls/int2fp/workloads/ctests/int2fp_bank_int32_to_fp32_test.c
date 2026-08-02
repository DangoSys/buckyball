#include "buckyball.h"
#include "int2fp_ref.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ELEMS_PER_ROW 4
#define NELEM (BANK_LINES * ELEMS_PER_ROW)
#define SCALE_1_0 0x3F800000U
#define SEED 0x11111111U

static int32_t input_i32[NELEM] __attribute__((aligned(64)));
static float output_fp32[NELEM] __attribute__((aligned(64)));
static uint32_t expected_bits[NELEM] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  srand(SEED);
  double scale = (double)bits_to_f32(SCALE_1_0);

  for (int i = 0; i < NELEM; ++i) {
    int32_t v = (int32_t)(((uint32_t)rand() << 16) ^ (uint32_t)rand());
    input_i32[i] = v;
    expected_bits[i] = int2fp_fp32(v, SCALE_1_0);
    output_fp32[i] = 0.0f;
  }

  const uint32_t src = 0;
  const uint32_t dst = 1;
  bb_mem_alloc(src, BANK_LINES, 1);
  bb_mem_alloc(dst, BANK_LINES, 1);
  bb_mvin((uintptr_t)input_i32, src, BANK_LINES, 1);
  bb_int2fp(src, dst, BANK_LINES, SCALE_1_0);
  bb_mvout((uintptr_t)output_fp32, dst, BANK_LINES, 1);
  bb_fence();

  int passed = 1;
  double max_abs = 0.0;
  double sum_abs = 0.0;
  double max_rel = 0.0;
  for (int i = 0; i < NELEM; ++i) {
    uint32_t got = f32_to_bits(output_fp32[i]);
    if (got != expected_bits[i]) {
      printf("MISMATCH at [%d]: got 0x%08X, expected 0x%08X\n", i, got,
             expected_bits[i]);
      passed = 0;
    }
    double ideal = (double)input_i32[i] * scale;
    double err = (double)output_fp32[i] - ideal;
    if (err < 0.0)
      err = -err;
    if (err > max_abs)
      max_abs = err;
    sum_abs += err;
    double denom = ideal < 0.0 ? -ideal : ideal;
    if (denom > 1.0e-12) {
      double rel = err / denom;
      if (rel > max_rel)
        max_rel = rel;
    }
  }

  printf("LOSS n=%d max_abs=%.6f mean_abs=%.6f max_rel=%.6f\n", NELEM, max_abs,
         sum_abs / (double)NELEM, max_rel);
  printf("Int2Fp bank INT32-to-FP32 %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
