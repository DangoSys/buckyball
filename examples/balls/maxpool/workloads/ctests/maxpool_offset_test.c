#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/maxpool.h>
#include <stdint.h>
#include <stdio.h>

enum { LANES = BANK_WIDTH / 8, ROW = 3 };
static int8_t input[(ROW + 1) * LANES] __attribute__((aligned(64)));
static int8_t output[LANES] __attribute__((aligned(64)));

int main(void) {
  for (int c = 0; c < LANES; ++c)
    input[ROW * LANES + c] = (int8_t)(c - LANES / 2);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, ROW + 1, 1);
  bb_maxpool(0, 1, 4, 1, 1, 1, 0, ROW, 0, 1, 0, 0);
  bb_mvout((uintptr_t)output, 1, 1, 1);
  bb_fence();
  for (int c = 0; c < LANES; ++c)
    if (output[c] != (int8_t)(c - LANES / 2)) {
      printf("maxpool offset mismatch c=%d\n", c);
      return 1;
    }
  printf("maxpool offset PASS\n");
  return 0;
}
