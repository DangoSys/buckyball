#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdio.h>

#define NUM_WINDOWS 4
#define KERNEL_ELEMS 4

/* A: 4x4 identity-ish in first 4 lanes; B: col0 = [1,2,3,4] */
static int8_t matrix_a[NUM_WINDOWS * 16] __attribute__((aligned(64)));
static int8_t matrix_b[KERNEL_ELEMS * 16] __attribute__((aligned(64)));
static int32_t zero[NUM_WINDOWS * 16] __attribute__((aligned(64)));
static int32_t actual[NUM_WINDOWS * 16] __attribute__((aligned(64)));
static const int32_t expected[NUM_WINDOWS] = {1, 2, 3, 4};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  for (int i = 0; i < NUM_WINDOWS * 16; ++i) {
    matrix_a[i] = 0;
    zero[i] = 0;
    actual[i] = 0;
  }
  for (int i = 0; i < KERNEL_ELEMS * 16; ++i)
    matrix_b[i] = 0;
  for (int r = 0; r < NUM_WINDOWS; ++r)
    matrix_a[r * 16 + r] = 1;
  for (int r = 0; r < KERNEL_ELEMS; ++r)
    matrix_b[r * 16] = (int8_t)(r + 1);

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  bb_mvin((uintptr_t)matrix_a, 0, NUM_WINDOWS, 1);
  bb_mvin((uintptr_t)matrix_b, 1, KERNEL_ELEMS, 1);
  bb_mvin((uintptr_t)zero, 2, NUM_WINDOWS, 1);
  bb_smatmul_os(0, 1, 2, NUM_WINDOWS, 1, KERNEL_ELEMS, 1, 1);
  bb_mvout((uintptr_t)actual, 2, NUM_WINDOWS, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);

  int passed = 1;
  for (int i = 0; i < NUM_WINDOWS; ++i) {
    if (actual[i * 16] != expected[i]) {
      printf("FAIL i=%d exp=%d got=%d\n", i, (int)expected[i],
             (int)actual[i * 16]);
      passed = 0;
    }
  }
  printf("quant_gemm %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
