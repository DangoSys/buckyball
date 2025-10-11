#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_SIZE 16 // 16 elements for simple test
#define VECLANE 16

static elem_t input_data[TEST_SIZE] __attribute__((aligned(16)));
static elem_t output_data[TEST_SIZE] __attribute__((aligned(16)));
static elem_t expected_data[TEST_SIZE] __attribute__((aligned(16)));

// Software Softmax implementation (simplified)
void sw_softmax(const elem_t *input, elem_t *output, int size) {
  // Find max
  elem_t max_val = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  // Compute exp(x - max) and sum
  int sum_exp = 0;
  int exp_vals[TEST_SIZE];
  for (int i = 0; i < size; i++) {
    int shifted = input[i] - max_val;
    // Simple approximation: exp(x) ≈ max(0, 1 + x) for small x
    // Scale by 256 for fixed point
    int exp_val;
    if (shifted < -16) {
      exp_val = 0;
    } else if (shifted > 16) {
      exp_val = 4096; // Large value
    } else {
      exp_val = 256 + (shifted << 4); // 256 + x*16
      if (exp_val < 0)
        exp_val = 0;
    }
    exp_vals[i] = exp_val;
    sum_exp += exp_val;
  }

  // Normalize
  if (sum_exp == 0)
    sum_exp = 1;
  for (int i = 0; i < size; i++) {
    // output = (exp_val * 256) / sum_exp
    int normalized = (exp_vals[i] * 256) / sum_exp;
    // Clamp to INT8 range
    if (normalized > 127)
      normalized = 127;
    if (normalized < -128)
      normalized = -128;
    output[i] = (elem_t)normalized;
  }
}

// Hardware Softmax function
void hw_softmax(const char *test_name, elem_t *input, elem_t *output, int iter,
                int dim_len, int batch, int log_mode) {
  uint32_t op1_bank = 0;
  uint32_t op1_addr = 0;
  uint32_t wr_bank = 1;
  uint32_t wr_addr = 0;
  uint32_t is_acc = 0; // Use SRAM mode

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
    int diff = abs(a[i] - b[i]);
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

// Test 1: Simple softmax with known values
int test_simple_softmax() {
  printf("Test 1: Simple Softmax (16 elements)\n");

  // Simple input: [0, 1, 2, 3, ..., 15]
  for (int i = 0; i < TEST_SIZE; i++) {
    input_data[i] = i;
  }

  // Compute expected output
  sw_softmax(input_data, expected_data, TEST_SIZE);

  // Run hardware Softmax
  hw_softmax("Simple", input_data, output_data, 1, TEST_SIZE, 1, 0);

  // Compare results (allow some tolerance due to approximation)
  if (compare_arrays_with_tolerance(output_data, expected_data, TEST_SIZE,
                                    10)) {
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
  for (int i = 0; i < TEST_SIZE; i++) {
    input_data[i] = 0;
  }

  // Compute expected output
  sw_softmax(input_data, expected_data, TEST_SIZE);

  // Run hardware Softmax
  hw_softmax("Zeros", input_data, output_data, 1, TEST_SIZE, 1, 0);

  // All outputs should be equal (uniform distribution)
  if (compare_arrays_with_tolerance(output_data, expected_data, TEST_SIZE,
                                    10)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test 3: One hot (one large value, rest zeros)
int test_one_hot() {
  printf("Test 3: One-Hot Distribution\n");

  // Set one value to maximum, rest to minimum
  for (int i = 0; i < TEST_SIZE; i++) {
    input_data[i] = (i == 8) ? 127 : -128;
  }

  // Compute expected output
  sw_softmax(input_data, expected_data, TEST_SIZE);

  // Run hardware Softmax
  hw_softmax("One-Hot", input_data, output_data, 1, TEST_SIZE, 1, 0);

  // Output at index 8 should be much larger than others
  if (compare_arrays_with_tolerance(output_data, expected_data, TEST_SIZE,
                                    20)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test 4: Random values
int test_random() {
  printf("Test 4: Random Values\n");

  // Generate random input data
  for (int i = 0; i < TEST_SIZE; i++) {
    input_data[i] = (rand() % 256) - 128;
  }

  // Compute expected output
  sw_softmax(input_data, expected_data, TEST_SIZE);

  // Run hardware Softmax
  hw_softmax("Random", input_data, output_data, 1, TEST_SIZE, 1, 0);

  // Compare results
  if (compare_arrays_with_tolerance(output_data, expected_data, TEST_SIZE,
                                    15)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test 5: Negative values
int test_negative() {
  printf("Test 5: All Negative Values\n");

  // All negative values
  for (int i = 0; i < TEST_SIZE; i++) {
    input_data[i] = -(i + 1);
  }

  // Compute expected output
  sw_softmax(input_data, expected_data, TEST_SIZE);

  // Run hardware Softmax
  hw_softmax("Negative", input_data, output_data, 1, TEST_SIZE, 1, 0);

  // Compare results
  if (compare_arrays_with_tolerance(output_data, expected_data, TEST_SIZE,
                                    10)) {
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
  passed += test_negative();

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
