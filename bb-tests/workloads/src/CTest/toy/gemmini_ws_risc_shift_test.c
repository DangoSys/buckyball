#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define DIM 16
#define SHIFT 4

static elem_t mat_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t mat_b[DIM * DIM] __attribute__((aligned(64)));
static elem_t mat_d[DIM * DIM] __attribute__((aligned(64)));
static result_t mat_c[DIM * DIM] __attribute__((aligned(64)));
static result_t expected[DIM * DIM] __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("=== Gemmini WS RISC in_shift Test ===\n");

  for (int i = 0; i < DIM * DIM; i++) {
    mat_a[i] = (elem_t)((i + 1) % 128);
    mat_b[i] = (elem_t)((2 * (i + 1)) % 128);
  }
  clear_u8_matrix(mat_d, DIM, DIM);
  cpu_matmul(mat_a, mat_b, expected, DIM, DIM, DIM);
  for (int i = 0; i < DIM * DIM; i++)
    expected[i] >>= SHIFT;

  uint32_t bank_w = 0, bank_a = 1, bank_d = 2, bank_c = 3;
  bb_mem_alloc(bank_w, 1, 1);
  bb_mem_alloc(bank_a, 1, 1);
  bb_mem_alloc(bank_d, 1, 1);
  bb_mem_alloc(bank_c, 1, 4);
  bb_mvin((uintptr_t)mat_b, bank_w, DIM, 1);
  bb_mvin((uintptr_t)mat_a, bank_a, DIM, 1);
  bb_mvin((uintptr_t)mat_d, bank_d, DIM, 1);
  bb_gemmini_config(1, 0, 0, 0, SHIFT);
  bb_gemmini_preload(bank_w, bank_c, DIM);
  bb_gemmini_compute_preloaded(bank_a, bank_d, bank_c, DIM);
  bb_mvout((uintptr_t)mat_c, bank_c, DIM, 1);
  bb_fence();

  if (compare_u32_matrices(mat_c, expected, DIM, DIM)) {
    printf("Gemmini WS RISC in_shift Test PASSED\n");
    return 0;
  }
  printf("Gemmini WS RISC in_shift Test FAILED\n");
  return 1;
}
