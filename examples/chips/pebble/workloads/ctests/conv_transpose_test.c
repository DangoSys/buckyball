#include "pebble.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#ifndef TEST_CASE_NAME
#define TEST_CASE_NAME "conv_6x6_k3"
#endif

enum { MAX_K = 7 * 7 };

static elem_t src[PEBBLE_INT8_LANES * PEBBLE_INT8_LANES]
    __attribute__((aligned(64)));
static elem_t actual[MAX_K * PEBBLE_INT8_LANES] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  const pebble_conv_test_case_t *t = pebble_find_conv_test_case(TEST_CASE_NAME);
  if (!t) {
    printf("unknown case %s\n", TEST_CASE_NAME);
    return 1;
  }

  int k_elems = t->kernel_h * t->kernel_w;
  if (k_elems > PEBBLE_INT8_LANES) {
    printf("k_elems=%d exceeds lanes=%d\n", k_elems, PEBBLE_INT8_LANES);
    return 1;
  }

  for (int i = 0; i < PEBBLE_INT8_LANES * PEBBLE_INT8_LANES; ++i)
    src[i] = 0;
  for (int i = 0; i < k_elems; ++i)
    src[i] = t->kernel[i];

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)src, 0, PEBBLE_INT8_LANES, 1);
  bb_transpose(0, 1, PEBBLE_INT8_LANES, 8);
  bb_mvout((uintptr_t)actual, 1, k_elems, 1);
  bb_fence();

  int passed = 1;
  for (int row = 0; row < k_elems; ++row) {
    for (int col = 0; col < PEBBLE_INT8_LANES; ++col) {
      elem_t exp = (col == 0) ? t->kernel[row] : 0;
      elem_t got = actual[row * PEBBLE_INT8_LANES + col];
      if (got != exp) {
        printf("FAIL transpose row=%d col=%d exp=%d got=%d\n", row, col,
               (int)exp, (int)got);
        passed = 0;
      }
    }
  }
  printf("conv_transpose [%s] %s\n", t->name, passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
