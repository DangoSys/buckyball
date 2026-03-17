#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define DIM 16

static elem_t mat_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t mat_b[DIM * DIM] __attribute__((aligned(64)));
static result_t mat_c[DIM * DIM] __attribute__((aligned(64)));
static result_t expected[DIM * DIM] __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("=== Gemmini OS CISC Basic Test ===\n");

  init_u8_random_matrix(mat_a, DIM, DIM, 42);
  init_u8_random_matrix(mat_b, DIM, DIM, 84);
  cpu_matmul(mat_a, mat_b, expected, DIM, DIM, DIM);

  bb_gemmini_config(0, 0, 0, 0, 0);
  bb_gemmini_loop_ws_config_bounds(1, 1, 1);
  bb_gemmini_loop_ws_config_addr_a((uintptr_t)mat_a);
  bb_gemmini_loop_ws_config_addr_b((uintptr_t)mat_b);
  bb_gemmini_loop_ws_config_addr_d(0);
  bb_gemmini_loop_ws_config_addr_c((uintptr_t)mat_c);
  bb_gemmini_loop_ws_config_strides_ab(DIM, DIM);
  bb_gemmini_loop_ws_config_strides_dc(0, DIM * 4);
  bb_gemmini_loop_ws(0, 1, 2, 1);
  bb_fence();

  if (compare_u32_matrices(mat_c, expected, DIM, DIM)) {
    printf("Gemmini OS CISC Basic Test PASSED\n");
    return 0;
  }
  printf("Gemmini OS CISC Basic Test FAILED\n");
  return 1;
}
