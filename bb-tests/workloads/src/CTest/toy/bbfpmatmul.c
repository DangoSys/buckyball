#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

void init_matrix(elem_t *matrix, int rows, int cols, int seed) {
  srand(seed);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] =
        rand() % 4; // Initialize with random values in the range [0, 127]
  }
}
void flip_matrix(elem_t *matrix, int rows, int cols) {
  for (int i = 0; i < rows / 2; i++) {
    for (int j = 0; j < cols; j++) {
      elem_t temp = matrix[i * cols + j];
      matrix[i * cols + j] = matrix[(rows - 1 - i) * cols + j];
      matrix[(rows - 1 - i) * cols + j] = temp;
    }
  }
}
// Test matrices
static elem_t input_matrix[DIM * DIM] __attribute__((aligned(64)));
static elem_t transposed_matrix[DIM * DIM] __attribute__((aligned(64)));
static elem_t weight_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_output_matrix[DIM * DIM] __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE); // Only allow specified hart to continue
#endif

  // Initialize weight matrix
  init_matrix(weight_matrix, DIM, DIM, 42);
  init_matrix(input_matrix, DIM, DIM, 51);

  // Clear output matrix
  memset(output_matrix, 0, sizeof(output_matrix));
  cpu_matmul(input_matrix, weight_matrix, expected_output_matrix, DIM, DIM,
             DIM);
  // print_matrix("Input", input_matrix, DIM, DIM);

  // Move input to scratchpad
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);

  bb_mvin((uintptr_t)weight_matrix, op1_bank_id, DIM, 1);
  bb_mvin((uintptr_t)input_matrix, op2_bank_id, DIM, 1);
  bb_fence();
  bb_bbfp_mul(op1_bank_id, op2_bank_id, acc_bank_id, DIM);
  bb_fence();
  // Move back from scratchpad to output
  bb_mvout((uintptr_t)output_matrix, op2_bank_id, DIM << 2, 1);
  bb_fence();
  if (compare_u32_matrices(output_matrix, expected_output_matrix, DIM, DIM)) {
    printf("Test passed!\n");
  } else {
    printf("Test failed!\n");
  }
  // print_matrix("Output", output_matrix, DIM, DIM);

#ifdef MULTICORE
  exit(0);
#endif
}
