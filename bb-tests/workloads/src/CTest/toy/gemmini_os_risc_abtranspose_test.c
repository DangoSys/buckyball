#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define DIM 16

static elem_t mat_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t mat_b[DIM * DIM] __attribute__((aligned(64)));
static elem_t mat_at[DIM * DIM] __attribute__((aligned(64)));
static elem_t mat_bt[DIM * DIM] __attribute__((aligned(64)));
static result_t mat_c[DIM * DIM] __attribute__((aligned(64)));
static result_t expected[DIM * DIM] __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("=== Gemmini OS RISC ab_transpose Test ===\n");

  init_u8_random_matrix(mat_a, DIM, DIM, 42);
  init_u8_random_matrix(mat_b, DIM, DIM, 84);
  transpose_u8_matrix(mat_a, mat_at, DIM, DIM);
  transpose_u8_matrix(mat_b, mat_bt, DIM, DIM);
  cpu_matmul(mat_at, mat_bt, expected, DIM, DIM, DIM);

  uint32_t bank_a = 0, bank_b = 1, bank_c = 2;
  bb_mem_alloc(bank_a, 1, 1);
  bb_mem_alloc(bank_b, 1, 1);
  bb_mem_alloc(bank_c, 1, 4);
  bb_mvin((uintptr_t)mat_a, bank_a, DIM, 1);
  bb_mvin((uintptr_t)mat_b, bank_b, DIM, 1);
  bb_gemmini_config(0, 0, 1, 1, 0);
  bb_gemmini_preload(bank_a, bank_c, DIM);
  bb_gemmini_compute_preloaded(bank_a, bank_b, bank_c, DIM);
  bb_mvout((uintptr_t)mat_c, bank_c, DIM, 1);
  bb_fence();

  if (compare_u32_matrices(mat_c, expected, DIM, DIM)) {
    printf("Test PASSED\n");
    return 0;
  }
  printf("Test FAILED\n");
  return 1;
}
