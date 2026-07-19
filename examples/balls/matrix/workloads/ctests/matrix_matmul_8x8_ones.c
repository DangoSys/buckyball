#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 8

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

void hw_matmul(elem_t *a, elem_t *b, result_t *c, int size) {
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  int acc_bank_id = 2;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);
  bb_mem_alloc(acc_bank_id, 1, 4);

  bb_mvin((uintptr_t)a, op1_bank_id, DIM, 1);
  bb_mvin((uintptr_t)b, op2_bank_id, DIM, 1);

  bb_matrix(op1_bank_id, op2_bank_id, acc_bank_id, size);
  bb_mvout((uintptr_t)c, acc_bank_id, size << 2, 1);
  bb_fence();
}

int run_test(elem_t *a, elem_t *b, int size) {
  cpu_matmul(a, b, expected_matrix, size, size, size);
  hw_matmul(a, b, output_matrix, size);
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_ones_matrix(input_matrix_a, DIM, DIM);
  init_ones_matrix(input_matrix_b, DIM, DIM);
  return run_test(input_matrix_a, input_matrix_b, DIM) ? 0 : 1;
#ifdef MULTICORE
  exit(0);
#endif
}
