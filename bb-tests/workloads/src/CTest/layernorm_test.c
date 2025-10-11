#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>

// Simple integer square root for baremetal (Babylonian method)
static inline int32_t int_sqrt(int32_t x) {
  if (x <= 0)
    return 1;

  int32_t guess = x >> 1;
  if (guess == 0)
    guess = 1;

  // Newton's method iterations
  for (int i = 0; i < 10; i++) {
    int32_t new_guess = (guess + x / guess) >> 1;
    if (new_guess == guess)
      break;
    guess = new_guess;
  }

  return guess;
}

#define TEST_BATCH 4 // Number of batches
#define NORM_DIM 4 // Normalization dimension in vectors (4 * 16 = 64 elements)
#define ELEM_PER_VEC 16 // Elements per vector
#define TOTAL_ELEMS (NORM_DIM * ELEM_PER_VEC)

static result_t input_data[TEST_BATCH][TOTAL_ELEMS]
    __attribute__((aligned(64)));
static result_t output_data[TEST_BATCH][TOTAL_ELEMS]
    __attribute__((aligned(64)));
static result_t expected_data[TEST_BATCH][TOTAL_ELEMS]
    __attribute__((aligned(64)));

// Software LayerNorm implementation (simplified, integer-based)
void sw_layernorm(const result_t *input, result_t *output, int N) {
  // Compute mean
  int64_t sum = 0;
  for (int i = 0; i < N; i++) {
    sum += input[i];
  }
  result_t mean = sum / N;

  // Compute variance
  int64_t var_sum = 0;
  for (int i = 0; i < N; i++) {
    int64_t diff = input[i] - mean;
    var_sum += diff * diff;
  }
  result_t variance = var_sum / N;

  // Compute std using integer sqrt
  // Add small epsilon (scaled integer) to avoid division by zero
  int32_t var_with_eps = variance + 1; // Simplified epsilon
  int32_t std = int_sqrt(var_with_eps);
  if (std == 0)
    std = 1; // Prevent division by zero

  // Normalize using integer arithmetic
  for (int i = 0; i < N; i++) {
    int32_t centered = input[i] - mean;
    output[i] = (centered * 100) / std; // Scale by 100 to preserve precision
  }
}

// Compute expected LayerNorm output for all batches
void compute_expected_layernorm() {
  for (int b = 0; b < TEST_BATCH; b++) {
    sw_layernorm(input_data[b], expected_data[b], TOTAL_ELEMS);
  }
}

// Hardware LayerNorm function (using ACC/INT32 mode)
void hw_layernorm(const char *test_name) {
  printf("Running hardware LayerNorm: %s\n", test_name);

  uint32_t op1_bank = 0; // ACC bank 0 for input
  uint32_t op1_addr = 0; // Starting address 0
  uint32_t wr_bank = 1;  // ACC bank 1 for output
  uint32_t wr_addr = 0;  // Starting address 0
  uint32_t is_acc = 1;   // Use ACC mode (INT32)

  // Flatten input data for DMA transfer
  result_t *flat_input = (result_t *)input_data;
  result_t *flat_output = (result_t *)output_data;

  // Note: In real implementation, we would use proper ACC mvin/mvout
  // For now, assuming data is already in ACC banks

  // Execute LayerNorm
  bb_layernorm_simple(op1_bank, op1_addr, wr_bank, wr_addr, TEST_BATCH, is_acc,
                      NORM_DIM);
  bb_fence();

  printf("  Hardware execution completed\n");
}

// Compare arrays with tolerance for fixed-point arithmetic
int compare_arrays_tolerance(const result_t *a, const result_t *b, int size,
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
  if (errors > 10) {
    printf("  ... and %d more errors\n", errors - 10);
  }
  return errors == 0;
}

// Test 1: Random data
int test_random_data() {
  printf("\nTest 1: Random data\n");

  // Generate random input data
  for (int b = 0; b < TEST_BATCH; b++) {
    for (int i = 0; i < TOTAL_ELEMS; i++) {
      input_data[b][i] = (rand() % 200) - 100; // Range: -100 to 99
    }
  }

  // Compute expected output
  compute_expected_layernorm();

  // Run hardware LayerNorm
  hw_layernorm("Random data");

  // Compare results (with tolerance for fixed-point arithmetic)
  int tolerance = 5; // Allow some error due to fixed-point computation
  int all_passed = 1;
  for (int b = 0; b < TEST_BATCH; b++) {
    printf("  Batch %d: ", b);
    if (compare_arrays_tolerance(output_data[b], expected_data[b], TOTAL_ELEMS,
                                 tolerance)) {
      printf("PASSED\n");
    } else {
      printf("FAILED\n");
      all_passed = 0;
    }
  }

  return all_passed;
}

// Test 2: Zeros
int test_zeros() {
  printf("\nTest 2: All zeros\n");

  // Clear arrays
  for (int b = 0; b < TEST_BATCH; b++) {
    for (int i = 0; i < TOTAL_ELEMS; i++) {
      input_data[b][i] = 0;
      expected_data[b][i] = 0;
    }
  }

  // Run hardware LayerNorm
  hw_layernorm("All zeros");

  // For all zeros, output should also be zeros
  int all_passed = 1;
  for (int b = 0; b < TEST_BATCH; b++) {
    printf("  Batch %d: ", b);
    if (compare_arrays_tolerance(output_data[b], expected_data[b], TOTAL_ELEMS,
                                 1)) {
      printf("PASSED\n");
    } else {
      printf("FAILED\n");
      all_passed = 0;
    }
  }

  return all_passed;
}

// Test 3: Constant value (should normalize to zero)
int test_constant() {
  printf("\nTest 3: Constant value\n");

  // Set all elements to same value
  for (int b = 0; b < TEST_BATCH; b++) {
    result_t val = 42 + b * 10;
    for (int i = 0; i < TOTAL_ELEMS; i++) {
      input_data[b][i] = val;
    }
  }

  // Compute expected output
  compute_expected_layernorm();

  // Run hardware LayerNorm
  hw_layernorm("Constant value");

  // For constant input, output should be close to zero (mean = input, var = 0)
  int all_passed = 1;
  for (int b = 0; b < TEST_BATCH; b++) {
    printf("  Batch %d: ", b);
    if (compare_arrays_tolerance(output_data[b], expected_data[b], TOTAL_ELEMS,
                                 5)) {
      printf("PASSED\n");
    } else {
      printf("FAILED\n");
      all_passed = 0;
    }
  }

  return all_passed;
}

// Test 4: Single batch
int test_single_batch() {
  printf("\nTest 4: Single batch\n");

  // Generate input for single batch
  for (int i = 0; i < TOTAL_ELEMS; i++) {
    input_data[0][i] = i - (TOTAL_ELEMS / 2); // Range: -32 to 31
  }

  // Compute expected output
  sw_layernorm(input_data[0], expected_data[0], TOTAL_ELEMS);

  // Run hardware LayerNorm with iter=1
  uint32_t op1_bank = 0;
  uint32_t op1_addr = 0;
  uint32_t wr_bank = 1;
  uint32_t wr_addr = 0;
  uint32_t is_acc = 1;

  bb_layernorm_simple(op1_bank, op1_addr, wr_bank, wr_addr, 1, is_acc,
                      NORM_DIM);
  bb_fence();

  // Compare results
  if (compare_arrays_tolerance(output_data[0], expected_data[0], TOTAL_ELEMS,
                               5)) {
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
  printf("LayerNorm Accelerator Test Suite\n");
  printf("===============================\n");
  printf("Config:\n");
  printf("  Batch size: %d\n", TEST_BATCH);
  printf("  Norm dimension: %d vectors (%d elements)\n", NORM_DIM, TOTAL_ELEMS);
  printf("  Data type: INT32 (ACC mode)\n");
  printf("===============================\n");

  int total_tests = 4;
  int passed = 0;

  passed += test_random_data();
  passed += test_zeros();
  passed += test_constant();
  passed += test_single_batch();

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
