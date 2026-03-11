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

void hw_gemmini_matmul(const char *test_name, elem_t *a, elem_t *b, result_t *c,
                       int size) {
  uint32_t op1_bank_id = 0; // A matrix
  uint32_t op2_bank_id = 1; // B matrix
  int acc_bank_id = 2;      // C output (accumulator width)

  bb_mem_alloc_private(op1_bank_id, 1, 1);
  bb_mem_alloc_private(op2_bank_id, 1, 1);
  bb_mem_alloc_private(acc_bank_id, 1, 4);

  // Load A and B into SRAM banks
  bb_mvin((uintptr_t)a, op1_bank_id, DIM, 1);
  bb_mvin((uintptr_t)b, op2_bank_id, DIM, 1);

  // Configure: OS mode, no activation, no transpose, no shift
  bb_gemmini_config(0, 0, 0, 0, 0);

  // Preload D=0 (zero bias), set output bank
  bb_gemmini_preload(op1_bank_id, acc_bank_id, size);

  // Compute: A * B + D(preloaded)
  bb_gemmini_compute_preloaded(op1_bank_id, op2_bank_id, acc_bank_id, size);

  // Read results back
  bb_mvout((uintptr_t)c, acc_bank_id, size << 2, 1);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  cpu_matmul(a, b, expected_matrix, size, size, size);
  hw_gemmini_matmul(test_name, a, b, output_matrix, size);

  if (compare_u32_matrices(output_matrix, expected_matrix, size, size)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_u8_random_matrix(input_matrix_a, DIM, DIM, 42);
  init_u8_random_matrix(input_matrix_b, DIM, DIM, 84);

  int passed =
      run_test("Gemmini OS Matmul", input_matrix_a, input_matrix_b, DIM);
  if (passed) {
    printf("Gemmini matmul test PASSED\n");
    return 0;
  } else {
    printf("Gemmini matmul test FAILED\n");
    return 1;
  }

#ifdef MULTICORE
  exit(0);
#endif
}
