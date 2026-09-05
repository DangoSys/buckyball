#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/quant.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define T 16
#define IN 4

static float img[2 * IN * IN] __attribute__((aligned(64)));
static int8_t actual[2 * IN * IN] __attribute__((aligned(64)));

int main(void) {
  int8_t expected[2 * IN * IN];
  for (int i = 0; i < IN * IN; ++i) {
    img[i] = ((i % 7) - 3) * 0.05f;
    img[IN * IN + i] = ((i % 11) - 5) * 4.0f;
  }

  float maxa = 0.f;
  for (int i = 0; i < 2 * IN * IN; ++i) {
    if (fabsf(img[i]) > maxa)
      maxa = fabsf(img[i]);
  }
  float da = maxa / 127.f;
  for (int i = 0; i < 2 * IN * IN; ++i) {
    int v = (int)(img[i] / da + (img[i] >= 0 ? 0.5f : -0.5f));
    expected[i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
  }

  bb_mem_alloc(0, IN * IN, 1);
  bb_mem_alloc(1, 2, 1);
  bb_mvin((uintptr_t)img, 0, (2 * IN * IN + 3) / 4, 1);
  bb_quant_f32_to_i8(0, 1, (2 * IN * IN + 3) / 4, 1.0f / da);
  bb_mvout((uintptr_t)actual, 1, 2, 1);
  bb_fence();

  for (int i = 0; i < 2 * IN * IN; ++i) {
    if (actual[i] != expected[i]) {
      printf("shared_da quant mismatch %d expected=%d actual=%d\n", i,
             expected[i], actual[i]);
      return 1;
    }
  }
  return printf("im2col_matadd_shared_da quant PASSED\n"), 0;
}
