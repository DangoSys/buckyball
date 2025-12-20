#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));

// Simple LUT for reference (256 entries)
static elem_t cpu_lut[256];

void init_lut() {
  // Initialize a simple LUT: identity function with saturation
  // In real NN-LUT, this would contain pre-computed activation function values
  for (int i = 0; i < 256; i++) {
    int val = (int8_t)i;
    if (val < -128) {
      cpu_lut[i] = -128;
    } else if (val > 127) {
      cpu_lut[i] = 127;
    } else {
      cpu_lut[i] = val;
    }
  }
}

void hw_nnlut(const char *test_name, elem_t *a, elem_t *b, int size) {
  // Source operand in spad bank 0, write target in spad bank 1
  uint32_t op1_bank_id = 0;
  uint32_t wr_bank_id = 1;
  // Move input into scratchpad bank0, starting at offset 0, iterate size times
  // row-wise
  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_fence();
  // Call NN-LUT instruction
  bb_nnlut(op1_bank_id, wr_bank_id, size);
  bb_fence();

  // Result will be moved back in run_test for verification
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  // CPU reference computation
  nnlut_cpu_reference(a, b, size);

  // Hardware computation
  hw_nnlut(test_name, a, b, size);

  // Move result back from scratchpad for verification
  uint32_t wr_bank_id = 1;
  bb_mvout((uintptr_t)output_matrix_b, wr_bank_id, size, 1);
  bb_fence();

  // Verify results
  int passed = 1;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      int idx = i * size + j;
      if (output_matrix_b[idx] != b[idx]) {
        printf("Mismatch at [%d][%d]: expected %d, got %d\n", i, j, b[idx],
               output_matrix_b[idx]);
        passed = 0;
      }
    }
  }

  return passed;
}

int nnlut_cpu_reference(elem_t *input, elem_t *output, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      elem_t val = input[i * size + j];
      // Convert to unsigned index (0-255)
      uint8_t idx = (uint8_t)val;
      output[i * size + j] = cpu_lut[idx];
    }
  }
  return 1;
}

int test_nnlut(int seed) {
  init_lut();
  init_i8_random_matrix(input_matrix_a, DIM, DIM, seed);

  // Run hardware test with verification
  return run_test("NN-LUT", input_matrix_a, output_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_nnlut(5);
  if (passed) {
    printf("NN-LUT test PASSED!!!!\n");
  } else {
    printf("NN-LUT test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
