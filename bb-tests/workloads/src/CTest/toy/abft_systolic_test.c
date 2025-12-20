#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_c[DIM * DIM] __attribute__((aligned(64)));

// CPU reference computation for matrix multiplication with ABFT
int abft_systolic_cpu_reference(elem_t *a, elem_t *b, elem_t *c, int size) {
  // Compute C = A * B
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      int32_t sum = 0;
      for (int k = 0; k < size; k++) {
        sum += (int32_t)a[i * size + k] * (int32_t)b[k * size + j];
      }
      // Clamp to int8_t range
      if (sum > 127) {
        c[i * size + j] = 127;
      } else if (sum < -128) {
        c[i * size + j] = -128;
      } else {
        c[i * size + j] = (elem_t)sum;
      }
    }
  }
  return 1;
}

void hw_abft_systolic(const char *test_name, elem_t *a, elem_t *b, elem_t *c,
                      int size) {
  // Matrix A in spad bank 0, Matrix B in spad bank 1, result in spad bank 2
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  uint32_t wr_bank_id = 2;

  // Move input matrices into scratchpad
  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_fence();
  bb_mvin((uintptr_t)b, op2_bank_id, size, 1);
  bb_fence();

  // Call ABFT systolic array instruction
  bb_abft_systolic(op1_bank_id, op2_bank_id, wr_bank_id, size);
  bb_fence();

  // Result will be moved back in run_test for verification
}

int run_test(const char *test_name, elem_t *a, elem_t *b, elem_t *c, int size) {
  // CPU reference computation
  abft_systolic_cpu_reference(a, b, c, size);

  // Hardware computation
  hw_abft_systolic(test_name, a, b, c, size);

  // Move result back from scratchpad for verification
  uint32_t wr_bank_id = 2;
  bb_mvout((uintptr_t)output_matrix_c, wr_bank_id, size, 1);
  bb_fence();

  // Verify results
  int passed = 1;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      int idx = i * size + j;
      if (output_matrix_c[idx] != c[idx]) {
        printf("Mismatch at [%d][%d]: expected %d, got %d\n", i, j, c[idx],
               output_matrix_c[idx]);
        passed = 0;
      }
    }
  }

  return passed;
}

int test_abft_systolic(int seed) {
  // Initialize input matrices with random values
  init_i8_random_matrix(input_matrix_a, DIM, DIM, seed);
  init_i8_random_matrix(input_matrix_b, DIM, DIM, seed + 1);

  // Run hardware test with verification
  return run_test("ABFT-Systolic", input_matrix_a, input_matrix_b,
                  output_matrix_c, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_abft_systolic(5);
  if (passed) {
    printf("ABFT-Systolic test PASSED!!!!\n");
  } else {
    printf("ABFT-Systolic test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
