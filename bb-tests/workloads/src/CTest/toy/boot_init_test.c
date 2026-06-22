#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

static void hw_boot_matmul(elem_t *a, elem_t *b, result_t *c, int size) {
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  int acc_bank_id = 2;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);
  bb_mem_alloc(acc_bank_id, 1, 4);

  bb_mvin((uintptr_t)a, op1_bank_id, DIM, 1);
  bb_mvin((uintptr_t)b, op2_bank_id, DIM, 1);

  bb_mul_warp16(op1_bank_id, op2_bank_id, acc_bank_id, size, 0);
  bb_mvout((uintptr_t)c, acc_bank_id, size << 2, 1);
  bb_fence();
}

static int run_boot_init_test(void) {
  init_sequence_matrix(input_matrix_a, DIM, DIM);
  init_sequence_matrix(input_matrix_b, DIM, DIM);

  cpu_matmul(input_matrix_a, input_matrix_b, expected_matrix, DIM, DIM, DIM);
  hw_boot_matmul(input_matrix_a, input_matrix_b, output_matrix, DIM);

  if (compare_u32_matrices(output_matrix, expected_matrix, DIM, DIM)) {
    printf("boot_init_test matmul PASSED\n");
    return 1;
  }

  printf("boot_init_test matmul FAILED\n");
  return 0;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = run_boot_init_test();

  if (passed) {
    printf("boot_init_test PASSED\n");
    return 0;
  }

  printf("boot_init_test FAILED\n");
  return 1;

#ifdef MULTICORE
  exit(0);
#endif
}
