#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * DIM] __attribute__((aligned(64)));

// Simple LIF neuron model for CPU reference
int snn_cpu_reference(elem_t *input, elem_t *output, int size,
                      uint8_t threshold, uint8_t leak_factor) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      int val = (int8_t)input[i * size + j];

      // Apply leak: multiply by leak_factor, then divide by 256
      int leaked = (val * (int)leak_factor) >> 8;

      // Fire condition: if leaked >= threshold, output threshold, else output
      // leaked Clamp to threshold range
      if (leaked >= (int)threshold) {
        output[i * size + j] = (elem_t)threshold;
      } else if (leaked < -(int)threshold) {
        output[i * size + j] = (elem_t)(-threshold);
      } else {
        output[i * size + j] = (elem_t)leaked;
      }
    }
  }
  return 1;
}

void hw_snn(const char *test_name, elem_t *a, elem_t *b, int size,
            uint8_t threshold, uint8_t leak_factor) {
  // Source operand in spad bank 0, write target in spad bank 1
  uint32_t op1_bank_id = 0;
  uint32_t wr_bank_id = 1;

  // Move input into scratchpad bank0, starting at offset 0, iterate size times
  // row-wise
  bb_mvin((uintptr_t)a, op1_bank_id, size, 1);
  bb_fence();
  // Call SNN instruction
  bb_snn(op1_bank_id, wr_bank_id, size, threshold, leak_factor);
  bb_fence();

  // Result will be moved back in run_test for verification
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size,
             uint8_t threshold, uint8_t leak_factor) {
  // CPU reference computation
  snn_cpu_reference(a, b, size, threshold, leak_factor);

  // Hardware computation
  hw_snn(test_name, a, b, size, threshold, leak_factor);

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

int test_snn(int seed) {
  // Initialize input matrix with random values
  init_i8_random_matrix(input_matrix_a, DIM, DIM, seed);

  // Test with default parameters: threshold=127, leak_factor=240
  uint8_t threshold = 127;
  uint8_t leak_factor = 240; // 240/256 ≈ 0.9375

  // Run hardware test with verification
  return run_test("SNN", input_matrix_a, output_matrix_b, DIM, threshold,
                  leak_factor);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_snn(5);
  if (passed) {
    printf("SNN test PASSED!!!!\n");
  } else {
    printf("SNN test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
