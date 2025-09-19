#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test matrices
static elem_t input_matrix[DIM * DIM] __attribute__((aligned(64)));
static elem_t weight_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));

// Utility function
void print_result_matrix(const char *name, result_t *matrix, int rows,
                         int cols) {
  printf("Matrix %s:\n", name);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%4d ", (int32_t)matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void init_matrixv2(elem_t *matrix, int rows, int cols, int seed, int value) {
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = value;
  }
}

int compare_matrices(result_t *a, result_t *b, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (a[i] != b[i]) {
      return 0; // Matrices are different
    }
  }
  return 1; // Matrices are the same
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE); // Only allow specified hart to continue
#endif

  // Initialize input matrix
  init_matrixv2(input_matrix, 16, 16, 42, 3);
  init_matrixv2(weight_matrix, 16, 16, 42, 2);
  // Clear output matrix
  memset(output_matrix, 0, sizeof(output_matrix));

  // print_matrix("Input", input_matrix, DIM, DIM);

  // Move input to scratchpad
  uint32_t op1_addr = spad_addr(0, 0); // spad0: 操作数A, 偏移0
  uint32_t op2_addr = spad_addr(1, 0); // spad1: 操作数B, 偏移0
  uint32_t wr_addr = spad_addr(2, 0);  // acc0: 写入累加器, 偏移0

  bb_mvin((uintptr_t)input_matrix, op1_addr, DIM);
  bb_mvin((uintptr_t)weight_matrix, op2_addr, DIM);
  bb_mvin((uintptr_t)output_matrix, wr_addr, DIM << 2);
  printf("Perform Matmul\n");
  bb_bbfp_mul(op1_addr, op2_addr, wr_addr, DIM);

  printf("change");
  bb_matmul_ws(wr_addr, op2_addr, wr_addr, 16);
  init_matrixv2(input_matrix, 16, 16, 42, 4);
  bb_matmul_ws(wr_addr, op2_addr, wr_addr, 16);
  printf("Matmul Done\n");
  bb_mvout(((uintptr_t)output_matrix), wr_addr, DIM << 2);

  print_result_matrix("Output", output_matrix, DIM, DIM);

#ifdef MULTICORE
  exit(0);
#endif
}
