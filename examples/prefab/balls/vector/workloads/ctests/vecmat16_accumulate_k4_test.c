#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/vecmat16.h>
#include <stdint.h>
#include <stdio.h>

// C = bias + A^T * B with a reduction depth of four rows, then a second
// instruction accumulating the same product again.
enum { DIM = 16, K = 4 };
static int8_t at[K * DIM] __attribute__((aligned(64)));
static int8_t b[K * DIM] __attribute__((aligned(64)));
static int32_t c[DIM * DIM] __attribute__((aligned(64)));
static int32_t expected[DIM * DIM];

int main(void) {
  for (int k = 0; k < K; ++k)
    for (int lane = 0; lane < DIM; ++lane) {
      at[k * DIM + lane] = (int8_t)((k * 37 + lane * 11) % 255 - 127);
      b[k * DIM + lane] = (int8_t)((k * 53 + lane * 29) % 255 - 128);
    }
  for (int i = 0; i < DIM; ++i)
    for (int j = 0; j < DIM; ++j) {
      int32_t bias = (i * DIM + j) * 1009 - 100000;
      int32_t product = 0;
      for (int k = 0; k < K; ++k)
        product += at[k * DIM + i] * b[k * DIM + j];
      c[i * DIM + j] = bias;
      expected[i * DIM + j] = bias + 2 * product;
    }
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 4);
  bb_mvin((uintptr_t)at, 1, K, 1);
  bb_mvin((uintptr_t)b, 2, K, 1);
  bb_mvin((uintptr_t)c, 3, DIM, 1);
  bb_vecmat16(1, 2, 3, K);
  bb_vecmat16(1, 2, 3, K);
  bb_mvout((uintptr_t)c, 3, DIM, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  for (int i = 0; i < DIM * DIM; ++i)
    if (c[i] != expected[i]) {
      printf("vecmat16 accumulate mismatch at %d: got %d expected %d\n", i,
             c[i], expected[i]);
      return 1;
    }
  printf("vecmat16_accumulate_k4 PASS\n");
  return 0;
}
