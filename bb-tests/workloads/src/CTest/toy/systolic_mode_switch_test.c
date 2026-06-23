#include "systolic_test_common.h"
#include <stdlib.h>

static elem_t input_matrix_a[SYSTOLIC_TEST_DIM * SYSTOLIC_TEST_DIM]
    __attribute__((aligned(64)));
static elem_t input_matrix_b[SYSTOLIC_TEST_DIM * SYSTOLIC_TEST_DIM]
    __attribute__((aligned(64)));
static elem_t
    packed_input_matrix_a[SYSTOLIC_TEST_DIM * SYSTOLIC_PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static elem_t
    packed_input_matrix_b[SYSTOLIC_TEST_DIM * SYSTOLIC_PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static result_t
    packed_output_matrix[SYSTOLIC_TEST_DIM * SYSTOLIC_PACKED_OUTPUT_COLS]
    __attribute__((aligned(64)));
static result_t output_matrix[SYSTOLIC_TEST_DIM * SYSTOLIC_TEST_DIM]
    __attribute__((aligned(64)));
static result_t expected_matrix[SYSTOLIC_TEST_DIM * SYSTOLIC_TEST_DIM]
    __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int os_first_passed = systolic_run_random_case(
      "systolic_mode_switch_os_first", 0, 123, 456, input_matrix_a,
      input_matrix_b, packed_input_matrix_a, packed_input_matrix_b,
      packed_output_matrix, output_matrix, expected_matrix);
  int ws_second_passed = systolic_run_random_case(
      "systolic_mode_switch_ws_second", 1, 789, 321, input_matrix_a,
      input_matrix_b, packed_input_matrix_a, packed_input_matrix_b,
      packed_output_matrix, output_matrix, expected_matrix);
  int os_third_passed = systolic_run_random_case(
      "systolic_mode_switch_os_third", 0, 654, 987, input_matrix_a,
      input_matrix_b, packed_input_matrix_a, packed_input_matrix_b,
      packed_output_matrix, output_matrix, expected_matrix);

  int rc = (os_first_passed && ws_second_passed && os_third_passed) ? 0 : 1;
#ifdef MULTICORE
  exit(rc);
#endif
  return rc;
}
