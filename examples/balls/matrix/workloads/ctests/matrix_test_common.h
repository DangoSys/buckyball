#ifndef MATRIX_TEST_COMMON_H
#define MATRIX_TEST_COMMON_H

#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define MATRIX_TEST_DIM 4
#define MATRIX_PACKED_INPUT_COLS (BANK_WIDTH / 8 / (int)sizeof(elem_t))
#define MATRIX_PACKED_OUTPUT_GROUPS 4
#define MATRIX_PACKED_OUTPUT_COLS                                              \
  (MATRIX_PACKED_OUTPUT_GROUPS * (BANK_WIDTH / 8 / (int)sizeof(result_t)))

static inline void matrix_pack_input_matrix(const elem_t *src, elem_t *dst) {
  for (int row = 0; row < MATRIX_TEST_DIM; ++row) {
    for (int col = 0; col < MATRIX_PACKED_INPUT_COLS; ++col) {
      dst[row * MATRIX_PACKED_INPUT_COLS + col] =
          (col < MATRIX_TEST_DIM) ? src[row * MATRIX_TEST_DIM + col] : 0;
    }
  }
}

static inline void matrix_unpack_output_matrix(const result_t *src,
                                               result_t *dst) {
  for (int row = 0; row < MATRIX_TEST_DIM; ++row) {
    for (int col = 0; col < MATRIX_TEST_DIM; ++col) {
      dst[row * MATRIX_TEST_DIM + col] =
          src[row * MATRIX_PACKED_OUTPUT_COLS + col * 4];
    }
  }
}

static inline void matrix_hw_matmul(const elem_t *a, const elem_t *b,
                                    result_t *c, int size, int ws_mode,
                                    elem_t *packed_a, elem_t *packed_b,
                                    result_t *packed_output) {
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  uint32_t acc_bank_id = 2;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);
  bb_mem_alloc(acc_bank_id, 1, 4);

  matrix_pack_input_matrix(a, packed_a);
  matrix_pack_input_matrix(b, packed_b);

  bb_mvin((uintptr_t)packed_a, op1_bank_id, MATRIX_TEST_DIM, 1);
  bb_mvin((uintptr_t)packed_b, op2_bank_id, MATRIX_TEST_DIM, 1);
  if (ws_mode) {
    bb_matrix_ws(op1_bank_id, op2_bank_id, acc_bank_id, size);
  } else {
    bb_matrix_os(op1_bank_id, op2_bank_id, acc_bank_id, size);
  }
  bb_mvout((uintptr_t)packed_output, acc_bank_id, size, 1);
  bb_fence();
  matrix_unpack_output_matrix(packed_output, c);
}

static inline int matrix_run_case(const char *test_name, const elem_t *a,
                                  const elem_t *b, int ws_mode,
                                  elem_t *packed_a, elem_t *packed_b,
                                  result_t *packed_output, result_t *output,
                                  result_t *expected) {
  clear_u32_matrix(output, MATRIX_TEST_DIM, MATRIX_TEST_DIM);
  clear_u32_matrix(expected, MATRIX_TEST_DIM, MATRIX_TEST_DIM);
  clear_u32_matrix(packed_output, MATRIX_TEST_DIM, MATRIX_PACKED_OUTPUT_COLS);

  cpu_matmul((elem_t *)a, (elem_t *)b, expected, MATRIX_TEST_DIM,
             MATRIX_TEST_DIM, MATRIX_TEST_DIM);
  matrix_hw_matmul(a, b, output, MATRIX_TEST_DIM, ws_mode, packed_a, packed_b,
                   packed_output);

  if (!compare_u32_matrices(output, expected, MATRIX_TEST_DIM,
                            MATRIX_TEST_DIM)) {
    printf("%s FAILED\n", test_name);
    return 0;
  }

  printf("%s PASSED\n", test_name);
  return 1;
}

static inline int matrix_run_random_case(const char *test_name, int ws_mode,
                                         int seed_a, int seed_b,
                                         elem_t *input_a, elem_t *input_b,
                                         elem_t *packed_a, elem_t *packed_b,
                                         result_t *packed_output,
                                         result_t *output, result_t *expected) {
  init_u8_random_matrix(input_a, MATRIX_TEST_DIM, MATRIX_TEST_DIM, seed_a);
  init_u8_random_matrix(input_b, MATRIX_TEST_DIM, MATRIX_TEST_DIM, seed_b);
  return matrix_run_case(test_name, input_a, input_b, ws_mode, packed_a,
                         packed_b, packed_output, output, expected);
}

#endif
