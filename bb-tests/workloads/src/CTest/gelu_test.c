#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_SIZE 64 // 64 vectors = 1024 elements

static elem_t input_data[TEST_SIZE * 16] __attribute__((aligned(16)));
static elem_t output_data[TEST_SIZE * 16] __attribute__((aligned(16)));
static elem_t expected_data[TEST_SIZE * 16] __attribute__((aligned(16)));

// Software GELU implementation (simplified piecewise linear approximation)
static elem_t sw_gelu(elem_t x) {
  int threshold = 3;
  if (x >= threshold) {
    return x;
  } else if (x <= -threshold) {
    return 0;
  } else if (x >= 0) {
    return x;
  } else {
    // x < 0: return x/2
    return x >> 1;
  }
}

// Compute expected GELU output
void compute_expected_gelu(const elem_t *input, elem_t *output, int size) {
  for (int i = 0; i < size; i++) {
    output[i] = sw_gelu(input[i]);
  }
}

// Hardware GELU function
void hw_gelu(const char *test_name, elem_t *input, elem_t *output, int iter) {
  uint32_t op1_addr = spad_addr(0, 0); // spad0: input data
  uint32_t wr_addr = spad_addr(1, 0);  // spad1: output data

  // Move input data to scratchpad
  bb_mvin((uintptr_t)input, op1_addr, iter, 1);
  bb_fence();

  // Execute GELU
  bb_gelu(op1_addr, wr_addr, iter);
  bb_fence();

  // Move output data from scratchpad
  bb_mvout((uintptr_t)output, wr_addr, iter);
  bb_fence();
}

// Compare arrays
int compare_arrays(const elem_t *a, const elem_t *b, int size) {
  int errors = 0;
  for (int i = 0; i < size; i++) {
    if (a[i] != b[i]) {
      if (errors < 10) {
        printf("  Mismatch at index %d: got %d, expected %d\n", i, a[i], b[i]);
      }
      errors++;
    }
  }
  return errors == 0;
}

// Test with random data
int test_random_data() {
  printf("Test 1: Random data\n");

  // Generate random input data
  for (int i = 0; i < TEST_SIZE * 16; i++) {
    input_data[i] = (rand() % 256) - 128; // Range: -128 to 127
  }

  // Compute expected output
  compute_expected_gelu(input_data, expected_data, TEST_SIZE * 16);

  // Run hardware GELU
  hw_gelu("Random data", input_data, output_data, TEST_SIZE);

  // Compare results
  if (compare_arrays(output_data, expected_data, TEST_SIZE * 16)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test with all zeros
int test_zeros() {
  printf("Test 2: All zeros\n");

  // Clear arrays
  for (int i = 0; i < TEST_SIZE * 16; i++) {
    input_data[i] = 0;
    expected_data[i] = 0;
  }

  // Run hardware GELU
  hw_gelu("All zeros", input_data, output_data, TEST_SIZE);

  // Compare results
  if (compare_arrays(output_data, expected_data, TEST_SIZE * 16)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test with positive values
int test_positive() {
  printf("Test 3: Positive values\n");

  // Generate positive input data
  for (int i = 0; i < TEST_SIZE * 16; i++) {
    input_data[i] = i % 127;
  }

  // Compute expected output
  compute_expected_gelu(input_data, expected_data, TEST_SIZE * 16);

  // Run hardware GELU
  hw_gelu("Positive values", input_data, output_data, TEST_SIZE);

  // Compare results
  if (compare_arrays(output_data, expected_data, TEST_SIZE * 16)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test with negative values
int test_negative() {
  printf("Test 4: Negative values\n");

  // Generate negative input data
  for (int i = 0; i < TEST_SIZE * 16; i++) {
    input_data[i] = -(i % 128);
  }

  // Compute expected output
  compute_expected_gelu(input_data, expected_data, TEST_SIZE * 16);

  // Run hardware GELU
  hw_gelu("Negative values", input_data, output_data, TEST_SIZE);

  // Compare results
  if (compare_arrays(output_data, expected_data, TEST_SIZE * 16)) {
    printf("  PASSED\n");
    return 1;
  } else {
    printf("  FAILED\n");
    return 0;
  }
}

// Test with boundary values
int test_boundary() {
  printf("Test 5: Boundary values\n");

  // Test extreme values
  for (int i = 0; i < 16; i++) {
    input_data[i] = 127;       // Max positive
    input_data[16 + i] = -128; // Max negative
    input_data[32 + i] = 0;    // Zero
    input_data[48 + i] = 3;    // Threshold
  }

  // Compute expected output
  compute_expected_gelu(input_data, expected_data, 64);

  // Run hardware GELU (4 vectors)
  hw_gelu("Boundary values", input_data, output_data, 4);

  // Compare results
  if (compare_arrays(output_data, expected_data, 64)) {
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
  printf("GELU Accelerator Test Suite\n");
  printf("===============================\n\n");

  int total_tests = 5;
  int passed = 0;

  passed += test_random_data();
  passed += test_zeros();
  passed += test_positive();
  passed += test_negative();
  passed += test_boundary();

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
