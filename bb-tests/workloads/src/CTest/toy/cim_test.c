#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

#define OP1_SIZE 64
#define OP2_SIZE 64
#define RESULT_SIZE 64

static elem_t operand1[OP1_SIZE] __attribute__((aligned(64)));
static elem_t operand2[OP2_SIZE] __attribute__((aligned(64)));
static elem_t result[RESULT_SIZE] __attribute__((aligned(64)));
static elem_t expected_result[RESULT_SIZE] __attribute__((aligned(64)));

// CPU reference computation for CIM operations
int cim_cpu_reference(elem_t *op1, elem_t *op2, elem_t *result, int rows,
                      int cols, int op_type) {
  // op_type: 0=matmul, 1=add, 2=mul
  if (op_type == 0) {
    // Matrix multiplication: result = op1 * op2
    // Assume op1 is rows x cols, op2 is cols x cols
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        int32_t sum = 0;
        for (int k = 0; k < cols; k++) {
          sum += (int32_t)op1[i * cols + k] * (int32_t)op2[k * cols + j];
        }
        // Clamp to int8_t range
        if (sum > 127) {
          result[i * cols + j] = 127;
        } else if (sum < -128) {
          result[i * cols + j] = -128;
        } else {
          result[i * cols + j] = (elem_t)sum;
        }
      }
    }
  } else if (op_type == 1) {
    // Element-wise addition
    for (int i = 0; i < rows * cols; i++) {
      int32_t sum = (int32_t)op1[i] + (int32_t)op2[i];
      if (sum > 127) {
        result[i] = 127;
      } else if (sum < -128) {
        result[i] = -128;
      } else {
        result[i] = (elem_t)sum;
      }
    }
  } else if (op_type == 2) {
    // Element-wise multiplication
    for (int i = 0; i < rows * cols; i++) {
      int32_t prod = (int32_t)op1[i] * (int32_t)op2[i];
      if (prod > 127) {
        result[i] = 127;
      } else if (prod < -128) {
        result[i] = -128;
      } else {
        result[i] = (elem_t)prod;
      }
    }
  }
  return 1;
}

void hw_cim(const char *test_name, elem_t *op1, elem_t *op2, elem_t *result,
            int rows, int cols, int op_type, int acc_bank_id) {
  // Operand 1 in spad bank 0, operand 2 in spad bank 1, result in spad bank 2
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  // Move operand 1 into scratchpad
  bb_mvin((uintptr_t)op1, op1_bank_id, OP1_SIZE, 1);
  bb_fence();

  // Move operand 2 into scratchpad
  bb_mvin((uintptr_t)op2, op2_bank_id, OP2_SIZE, 1);
  bb_fence();

  // Call CIM instruction
  // iter is the number of iterations (simplified: use rows*cols for now)
  uint32_t iter = rows * cols;
  bb_cim(op1_bank_id, op2_bank_id, acc_bank_id, iter, rows, cols, op_type);
  bb_fence();

  // Result will be moved back in run_test for verification
}

int run_test(const char *test_name, elem_t *op1, elem_t *op2, elem_t *result,
             int rows, int cols, int op_type) {
  // CPU reference computation
  cim_cpu_reference(op1, op2, expected_result, rows, cols, op_type);

  // Allocate accumulator bank
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);

  // Hardware computation
  hw_cim(test_name, op1, op2, result, rows, cols, op_type, acc_bank_id);

  // Move result back from scratchpad for verification
  bb_mvout((uintptr_t)result, acc_bank_id, RESULT_SIZE, 1);
  bb_fence();

  // Verify results
  int passed = 1;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int idx = i * cols + j;
      if (result[idx] != expected_result[idx]) {
        printf("Mismatch at [%d][%d]: expected %d, got %d\n", i, j,
               expected_result[idx], result[idx]);
        passed = 0;
      }
    }
  }

  return passed;
}

int test_cim(int seed) {
  // Initialize operands with random values (8x8 matrices)
  int rows = 8;
  int cols = 8;
  for (int i = 0; i < OP1_SIZE; i++) {
    operand1[i] = (elem_t)(rand() % 256 - 128);
  }
  for (int i = 0; i < OP2_SIZE; i++) {
    operand2[i] = (elem_t)(rand() % 256 - 128);
  }

  // Test matrix multiplication (op_type = 0)
  int passed =
      run_test("CIM-MATMUL", operand1, operand2, result, rows, cols, 0);
  if (!passed) {
    return 0;
  }

  // Test element-wise addition (op_type = 1)
  passed = run_test("CIM-ADD", operand1, operand2, result, rows, cols, 1);
  if (!passed) {
    return 0;
  }

  // Test element-wise multiplication (op_type = 2)
  passed = run_test("CIM-MUL", operand1, operand2, result, rows, cols, 2);
  if (!passed) {
    return 0;
  }

  return passed;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_cim(5);
  if (passed) {
    printf("CIM test PASSED!!!!\n");
  } else {
    printf("CIM test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
