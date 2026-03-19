#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define DIM 16

static elem_t mat_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t mat_a_t[DIM * DIM] __attribute__((aligned(64)));
static elem_t mat_b[DIM * DIM] __attribute__((aligned(64)));
static elem_t zeros[DIM * DIM] __attribute__((aligned(64)));
static result_t mat_c[DIM * DIM] __attribute__((aligned(64)));
static result_t expected[DIM * DIM] __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  printf("=== Gemmini OS RISC Basic Test ===\n");

  for (int i = 0; i < DIM * DIM; i++) {
    mat_a[i] = (elem_t)((i + 1) % 128);
    mat_b[i] = (elem_t)((2 * (i + 1)) % 128);
  }
  transpose_u8_matrix(mat_a, mat_a_t, DIM, DIM);
  // Hardware (transposer disabled) computes mat_a_t^T * mat_b = mat_a * mat_b
  cpu_matmul(mat_a, mat_b, expected, DIM, DIM, DIM);

  uint32_t bank_a = 0, bank_b = 1, bank_c = 2, bank_d_zeros = 3;
  bb_mem_alloc(bank_a, 1, 1);
  bb_mem_alloc(bank_b, 1, 1);
  bb_mem_alloc(bank_c, 1, 4);
  bb_mem_alloc(bank_d_zeros, 1, 1);
  bb_mvin((uintptr_t)mat_a_t, bank_a, DIM, 1);
  bb_mvin((uintptr_t)mat_b, bank_b, DIM, 1);
  bb_mvin((uintptr_t)zeros, bank_d_zeros, DIM, 1);
  bb_gemmini_config(0, 0, 0, 0, 0);
  /* Preload D from zeros bank so C = A*B + D = A*B (not A*B + A) */
  bb_gemmini_preload(bank_d_zeros, bank_c, DIM);
  bb_gemmini_compute_preloaded(bank_a, bank_b, bank_c, DIM);
  bb_mvout((uintptr_t)mat_c, bank_c, DIM, 1);
  bb_fence();

  if (compare_u32_matrices(mat_c, expected, DIM, DIM)) {
    printf("Gemmini OS RISC Basic Test PASSED\n");
    return 0;
  }

  printf("got mat_c: ");
  for (int i = 0; i < DIM; i++) {
    printf("%d ", mat_c[i]);
  }
  printf("\n");
  // for (int i = 0; i < DIM; i++) {
  //   for (int j = 0; j < DIM; j++) {
  //     printf("%d ", mat_c[i * DIM + j]);
  //   }
  //   printf("\n");
  // }

  printf("exp row0: ");
  for (int i = 0; i < DIM; i++) {
    printf("%d ", expected[i]);
  }
  printf("\n");
  printf("Gemmini OS RISC Basic Test FAILED\n");
  return 1;
}
