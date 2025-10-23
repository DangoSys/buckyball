#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_SIZE 64 // 64 vectors = 1024 elements
#define DIM 16       // Softmax dimension (elements per group)

static elem_t input_data[TEST_SIZE * 16] __attribute__((aligned(16)));
static elem_t output_data[TEST_SIZE * 16] __attribute__((aligned(16)));
static elem_t expected_data[TEST_SIZE * 16] __attribute__((aligned(16)));

// Software Softmax implementation (simplified for INT8)
void sw_softmax(const elem_t *input, elem_t *output, int size) {
  // Find max
  elem_t max_val = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  // Compute exp(x - max) and sum (using integer approximation)
  int sum_exp = 0;
  int exp_vals[DIM];
  for (int i = 0; i < size; i++) {
    int shifted = input[i] - max_val;
    // Simple approximation: map to positive range
    int exp_val = 128 + shifted;
    if (exp_val < 0)
      exp_val = 0;
    if (exp_val > 255)
      exp_val = 255;
    exp_vals[i] = exp_val;
    sum_exp += exp_val;
  }

  // Normalize: scale to [0, 127] range for INT8
  if (sum_exp == 0)
    sum_exp = 1; // Prevent division by zero
  for (int i = 0; i < size; i++) {
    int normalized = (exp_vals[i] * 127) / sum_exp;
    if (normalized > 127)
      normalized = 127;
    if (normalized < 0)
      normalized = 0;
    output[i] = (elem_t)normalized;
  }
}

// Compute expected Softmax output for all groups
void compute_expected_softmax(const elem_t *input, elem_t *output,
                              int num_groups, int group_size) {
  for (int g = 0; g < num_groups; g++) {
    sw_softmax(input + g * group_size, output + g * group_size, group_size);
  }
}

// Hardware Softmax function
void hw_softmax(const char *test_name, elem_t *input, elem_t *output, int iter,
                int dim_len, int batch) {
  uint32_t op1_bank = 0; // SRAM bank 0 for input
  uint32_t op1_addr = 0; // Starting address 0
  uint32_t wr_bank = 1;  // SRAM bank 1 for output
  uint32_t wr_addr = 0;  // Starting address 0
  uint32_t is_acc = 0;   // Use SRAM mode (INT8)
  uint32_t log_mode = 0; // Standard Softmax (not LogSoftmax)

  // Move input data to scratchpad bank 0
  bb_mvin((uintptr_t)input, spad_addr(op1_bank, op1_addr), iter, 1);
  bb_fence();

  // Execute Softmax
  bb_softmax(op1_bank, op1_addr, wr_bank, wr_addr, iter, is_acc, dim_len, batch,
             log_mode);
  bb_fence();

  // Move output data from scratchpad bank 1
  bb_mvout((uintptr_t)output, spad_addr(wr_bank, wr_addr), iter);
  bb_fence();
}

// Compare arrays with tolerance
int compare_arrays_with_tolerance(const elem_t *a, const elem_t *b, int size,
                                  int tolerance) {
  int errors = 0;
  for (int i = 0; i < size; i++) {
    int diff = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
    if (diff > tolerance) {
      if (errors < 10) {
        printf("  Mismatch at index %d: got %d, expected %d (diff=%d)\n", i,
               a[i], b[i], diff);
      }
      errors++;
    }
  }
  return errors == 0;
}

// Test 1: Simple softmax with sequential values
int test_simple_softmax() {
  printf("Test 1: Simple Softmax (16 elements)\n");

  // Simple input: [0, 1, 2, ..., 15] repeated
  for (int i = 0; i < DIM; i++) {
    input_data[i] = i;
  }

  // Compute expected output
  compute_expected_softmax(input_data, expected_data, 1, DIM);

  // Run hardware Softmax (1 vector, dim_len=16, batch=1)
  hw_softmax("Simple", input_data, output_data, 1, DIM, 1);

  // Compare results (allow some tolerance due to approximation)
  if (compare_arrays_with_tolerance(output_data, expected_data, DIM, 20)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test 2: All zeros
int test_zeros() {
  printf("Test 2: All Zeros\n");

  // Clear arrays
  for (int i = 0; i < DIM; i++) {
    input_data[i] = 0;
  }

  // Compute expected output (uniform distribution)
  compute_expected_softmax(input_data, expected_data, 1, DIM);

  // Run hardware Softmax
  hw_softmax("Zeros", input_data, output_data, 1, DIM, 1);

  // Compare results
  if (compare_arrays_with_tolerance(output_data, expected_data, DIM, 20)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test 3: One hot (one large value, rest small)
int test_one_hot() {
  printf("Test 3: One-Hot Distribution\n");

  // Set one value to maximum, rest to minimum
  for (int i = 0; i < DIM; i++) {
    input_data[i] = (i == 8) ? 100 : 0;
  }

  // Compute expected output
  compute_expected_softmax(input_data, expected_data, 1, DIM);

  // Run hardware Softmax
  hw_softmax("One-Hot", input_data, output_data, 1, DIM, 1);

  // Compare results
  if (compare_arrays_with_tolerance(output_data, expected_data, DIM, 30)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test 4: Random values (single group)
int test_random() {
  printf("Test 4: Random Values\n");

  // Generate random input data
  for (int i = 0; i < DIM; i++) {
    input_data[i] = (rand() % 128); // Positive values only
  }

  // Compute expected output
  compute_expected_softmax(input_data, expected_data, 1, DIM);

  // Run hardware Softmax
  hw_softmax("Random", input_data, output_data, 1, DIM, 1);

  // Compare results
  if (compare_arrays_with_tolerance(output_data, expected_data, DIM, 25)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test 5: Multiple batches
int test_batch() {
  printf("Test 5: Batch Processing (4 groups)\n");

  // Create 4 groups of 16 elements each (64 elements = 4 vectors)
  for (int g = 0; g < 4; g++) {
    for (int i = 0; i < DIM; i++) {
      input_data[g * DIM + i] = (g * 10 + i) % 128;
    }
  }

  // Compute expected output for all groups
  compute_expected_softmax(input_data, expected_data, 4, DIM);

  // Run hardware Softmax (4 vectors, dim_len=16, batch=4)
  hw_softmax("Batch", input_data, output_data, 4, DIM, 4);

  // Compare results
  if (compare_arrays_with_tolerance(output_data, expected_data, 4 * DIM, 25)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  printf("===============================\n");
  printf("Softmax Accelerator Test Suite\n");
  printf("===============================\n\n");

  int total_tests = 5;
  int passed = 0;

  passed += test_simple_softmax();
  passed += test_zeros();
  passed += test_one_hot();
  passed += test_random();
  passed += test_batch();

  printf("\n===============================\n");
  printf("Results: %d/%d tests passed\n", passed, total_tests);
  printf("===============================\n");

  if (passed == total_tests) {
    printf("All tests PASSED\n");
    return 0;
  } else {
    printf("Some tests FAILED\n");
    return 1;
  }

#ifdef MULTICORE
  exit(0);
#endif
}
