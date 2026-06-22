#include "systolic_test_common.h"
#include <stdlib.h>

static elem_t probe_input_matrix_a[SYSTOLIC_TEST_DIM * SYSTOLIC_PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static elem_t probe_input_matrix_b[SYSTOLIC_TEST_DIM * SYSTOLIC_PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static result_t probe_output_matrix[SYSTOLIC_TEST_DIM * SYSTOLIC_PACKED_OUTPUT_COLS]
    __attribute__((aligned(64)));

static void init_probe_inputs(void) {
  clear_u8_matrix(probe_input_matrix_a, SYSTOLIC_TEST_DIM,
                  SYSTOLIC_PACKED_INPUT_COLS);
  clear_u8_matrix(probe_input_matrix_b, SYSTOLIC_TEST_DIM,
                  SYSTOLIC_PACKED_INPUT_COLS);

  for (int row = 0; row < SYSTOLIC_TEST_DIM; ++row) {
    for (int col = 0; col < SYSTOLIC_TEST_DIM; ++col) {
      probe_input_matrix_a[row * SYSTOLIC_PACKED_INPUT_COLS + col] =
          (elem_t)(row * SYSTOLIC_TEST_DIM + col + 1);
      probe_input_matrix_b[row * SYSTOLIC_PACKED_INPUT_COLS + col] =
          (elem_t)((row == col) ? 1 : 0);
    }
  }
}

static void init_probe_banks(void) {
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  uint32_t acc_bank_id = 2;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);
  bb_mem_alloc(acc_bank_id, 1, 4);
  bb_mvin((uintptr_t)probe_input_matrix_a, op1_bank_id, SYSTOLIC_TEST_DIM, 1);
  bb_mvin((uintptr_t)probe_input_matrix_b, op2_bank_id, SYSTOLIC_TEST_DIM, 1);
}

__attribute__((noinline)) static void issue_systolic_os_probe(void) {
  bb_BFP_OS(0, 1, 2, SYSTOLIC_TEST_DIM);
}

__attribute__((noinline)) static void issue_systolic_ws_probe(void) {
  bb_BFP_WS(0, 1, 2, SYSTOLIC_TEST_DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  init_probe_inputs();
  init_probe_banks();
  clear_u32_matrix(probe_output_matrix, SYSTOLIC_TEST_DIM,
                   SYSTOLIC_PACKED_OUTPUT_COLS);

  issue_systolic_os_probe();
  issue_systolic_ws_probe();

  bb_mvout((uintptr_t)probe_output_matrix, 2, SYSTOLIC_TEST_DIM, 1);
  bb_fence();

  printf("systolic_instruction_probe PASSED\n");

  int rc = 0;
#ifdef MULTICORE
  exit(rc);
#endif
  return rc;
}
