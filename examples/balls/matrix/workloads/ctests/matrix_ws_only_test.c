#include "matrix_test_common.h"
#include <stdlib.h>

static elem_t input_matrix_a[MATRIX_TEST_DIM * MATRIX_TEST_DIM]
    __attribute__((aligned(64)));
static elem_t input_matrix_b[MATRIX_TEST_DIM * MATRIX_TEST_DIM]
    __attribute__((aligned(64)));
static elem_t packed_input_matrix_a[MATRIX_TEST_DIM * MATRIX_PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static elem_t packed_input_matrix_b[MATRIX_TEST_DIM * MATRIX_PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static result_t
    packed_output_matrix[MATRIX_TEST_DIM * MATRIX_PACKED_OUTPUT_COLS]
    __attribute__((aligned(64)));
static result_t output_matrix[MATRIX_TEST_DIM * MATRIX_TEST_DIM]
    __attribute__((aligned(64)));
static result_t expected_matrix[MATRIX_TEST_DIM * MATRIX_TEST_DIM]
    __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = matrix_run_random_case(
      "matrix_ws_only_random", 1, 123, 456, input_matrix_a, input_matrix_b,
      packed_input_matrix_a, packed_input_matrix_b, packed_output_matrix,
      output_matrix, expected_matrix);

  int rc = passed ? 0 : 1;
#ifdef MULTICORE
  exit(rc);
#endif
  return rc;
}
