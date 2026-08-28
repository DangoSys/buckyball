#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/transpose.h>
#include <stdio.h>

#define KERNEL_ELEMS 9
#define LANES 16

static const int8_t kernel[KERNEL_ELEMS] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
static int8_t src[LANES * LANES] __attribute__((aligned(64)));
static int8_t actual[KERNEL_ELEMS * LANES] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  for (int i = 0; i < LANES * LANES; ++i)
    src[i] = 0;
  for (int i = 0; i < KERNEL_ELEMS; ++i)
    src[i] = kernel[i];
  bb_dma_touch(actual, sizeof(actual));

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)src, 0, LANES, 1);
  bb_transpose(0, 1, LANES, 8);
  bb_mvout((uintptr_t)actual, 1, KERNEL_ELEMS, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);

  int passed = 1;
  for (int row = 0; row < KERNEL_ELEMS; ++row) {
    for (int col = 0; col < LANES; ++col) {
      int8_t exp = (col == 0) ? kernel[row] : 0;
      if (actual[row * LANES + col] != exp) {
        printf("FAIL row=%d col=%d\n", row, col);
        passed = 0;
      }
    }
  }
  printf("quant_transpose %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
