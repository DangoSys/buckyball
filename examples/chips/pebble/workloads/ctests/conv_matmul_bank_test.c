#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <params.h>
#include <stdio.h>

/* One bank of A rows. */
#define M BANK_LINES
#define N 1
#define K 16
#define SEED_A 0xC3
#define SEED_B 0xC4

static elem_t a[M * K] __attribute__((aligned(64)));
static elem_t b[K * 16] __attribute__((aligned(64)));
static result_t zero[M * 16] __attribute__((aligned(64)));
static result_t actual[M * 16] __attribute__((aligned(64)));
static result_t expected[M];

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_i8_random_matrix(a, M, K, SEED_A);
  clear_i8_matrix(b, K, 16);
  init_i8_random_matrix(b, K, 1, SEED_B);
  for (int k = K - 1; k >= 0; --k) {
    elem_t v = b[k];
    b[k] = 0;
    b[k * 16] = v;
  }
  clear_u32_matrix(zero, M, 16);
  for (int i = 0; i < M; ++i) {
    result_t s = 0;
    for (int k = 0; k < K; ++k)
      s += (result_t)a[i * K + k] * (result_t)b[k * 16];
    expected[i] = s;
  }

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  bb_mvin((uintptr_t)a, 0, M, 1);
  bb_mvin((uintptr_t)b, 1, K, 1);
  bb_mvin((uintptr_t)zero, 2, M, 1);
  bb_smatmul_ws(0, 1, 2, M, N, K);
  bb_mvout((uintptr_t)actual, 2, M, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);

  int passed = 1;
  for (int i = 0; i < M; ++i) {
    if (actual[i * 16] != expected[i]) {
      printf("FAIL i=%d exp=%d got=%d\n", i, (int)expected[i],
             (int)actual[i * 16]);
      passed = 0;
      break;
    }
  }
  printf("conv_matmul_bank %s\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
