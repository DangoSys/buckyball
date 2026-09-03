#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/quant.h>
#include <stdint.h>
#include <stdio.h>

static float input[16] __attribute__((aligned(64))) = {
    -300.0f, -255.0f, -5.0f,  -3.0f,  -1.0f, 0.0f, 1.0f,  3.0f,
    5.0f,    254.0f,  255.0f, 300.0f, 2.0f,  6.0f, 10.0f, 14.0f,
};
static const int8_t expected[16] = {
    -128, -128, -2, -2, 0, 0, 0, 2, 2, 127, 127, 127, 1, 3, 5, 7,
};
static int8_t actual[16] __attribute__((aligned(64)));

int main(void) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, 4, 1);
  bb_quant_f32_to_i8(0, 1, 4, 0.5f);
  bb_mvout((uintptr_t)actual, 1, 1, 1);
  bb_fence();

  for (int i = 0; i < 16; ++i) {
    if (actual[i] != expected[i]) {
      printf("quant_f32_to_i8 FAIL i=%d expected=%d actual=%d\n", i,
             expected[i], actual[i]);
      return 1;
    }
  }
  printf("quant_f32_to_i8 PASS\n");
  return 0;
}
