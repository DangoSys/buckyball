#include "buckyball.h"
#include "fp2int_ref.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ELEMS_PER_ROW 16
#define NELEM (BANK_LINES * ELEMS_PER_ROW)
#define SCALE_1_0 0x3F800000U
#define SEED 0xC3C3C3C3U

static float input_fp32[NELEM] __attribute__((aligned(64)));
static int8_t output_i8[NELEM] __attribute__((aligned(64)));
static int8_t expected_i8[NELEM] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  srand(SEED);
  double scale = (double)bits_to_f32(SCALE_1_0);

  for (int i = 0; i < NELEM; ++i) {
    float in = (float)((rand() % 65536) - 32768) / 64.0f;
    input_fp32[i] = in;
    expected_i8[i] = fp2int_i8(f32_to_bits(in), SCALE_1_0);
    output_i8[i] = (int8_t)0x5a;
  }

  const uint32_t src = 0;
  const uint32_t dst = 1;
  bb_mem_alloc(src, BANK_LINES, 4);
  bb_mem_alloc(dst, BANK_LINES, 1);
  bb_mvin((uintptr_t)input_fp32, src, BANK_LINES, 1);
  bb_fp2int(src, dst, BANK_LINES, SCALE_1_0);
  bb_mvout((uintptr_t)output_i8, dst, BANK_LINES, 1);
  bb_fence();

  int passed = 1;
  double max_abs = 0.0;
  double sum_abs = 0.0;
  double max_rel = 0.0;
  for (int i = 0; i < NELEM; ++i) {
    if (output_i8[i] != expected_i8[i]) {
      printf("MISMATCH at [%d]: got %d, expected %d\n", i, output_i8[i],
             expected_i8[i]);
      passed = 0;
    }
    double ideal = (double)input_fp32[i] * scale;
    double err = (double)output_i8[i] - ideal;
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
  printf("Fp2Int bank FP32-to-INT8 %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
