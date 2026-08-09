#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <bbhw/mem/params.h>
#include <stdio.h>

/* Closer to MobileNet tile epilogue: compute, mvout, fence, bank
 * release/realloc, second mvout, fence. Hang was after ~24 tiles. */
#define M 16
#define N 1
#define K 16
#define LOOPS 24

static elem_t a[M * K] __attribute__((aligned(64)));
static elem_t b[K * 16] __attribute__((aligned(64)));
static result_t zero[M * 16] __attribute__((aligned(64)));
static result_t out[M * 16] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_i8_random_matrix(a, M, K, 0x11);
  clear_i8_matrix(b, K, 16);
  init_i8_random_matrix(b, K, 1, 0x22);
  for (int k = K - 1; k >= 0; --k) {
    elem_t v = b[k];
    b[k] = 0;
    b[k * 16] = v;
  }
  clear_u32_matrix(zero, M, 16);
  clear_u32_matrix(out, M, 16);

  printf("mvout_fence_loop start loops=%d\n", LOOPS);
  for (int i = 0; i < LOOPS; ++i) {
    bb_mem_alloc(0, 1, 1);
    bb_mem_alloc(1, 1, 1);
    bb_mem_alloc(2, 1, 4);
    bb_mvin((uintptr_t)a, 0, M, 1);
    bb_mvin((uintptr_t)b, 1, K, 1);
    bb_mvin((uintptr_t)zero, 2, M, 1);
    bb_matrix_mnk(0, 1, 2, M, N, K);
    bb_mvout((uintptr_t)out, 2, M, 1);
    bb_fence();

    bb_mem_release(2);
    bb_mem_alloc(2, 1, 4);
    bb_mvin((uintptr_t)zero, 2, M, 1);
    bb_mvout((uintptr_t)out, 2, M, 1);
    bb_fence();

    bb_mem_release(0);
    bb_mem_release(1);
    bb_mem_release(2);
    printf("mvout_fence_loop iter %d done\n", i);
  }
  printf("mvout_fence_loop PASS\n");
  return 0;
}
