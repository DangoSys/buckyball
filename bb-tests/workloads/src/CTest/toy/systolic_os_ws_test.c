#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 4
#define PACKED_INPUT_COLS (BANK_WIDTH / 8 / (int)sizeof(elem_t))
#define PACKED_OUTPUT_GROUPS 4
#define PACKED_OUTPUT_COLS \
  (PACKED_OUTPUT_GROUPS * (BANK_WIDTH / 8 / (int)sizeof(result_t)))
static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static elem_t packed_input_matrix_a[DIM * PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static elem_t packed_input_matrix_b[DIM * PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static result_t packed_output_matrix[DIM * PACKED_OUTPUT_COLS]
    __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));

static void pack_input_matrix(const elem_t *src, elem_t *dst) {
  for (int row = 0; row < DIM; ++row) {
    for (int col = 0; col < PACKED_INPUT_COLS; ++col) {
      dst[row * PACKED_INPUT_COLS + col] = (col < DIM) ? src[row * DIM + col] : 0;
    }
  }
}

static void unpack_output_matrix(const result_t *src, result_t *dst) {
  for (int row = 0; row < DIM; ++row) {
    for (int col = 0; col < DIM; ++col) {
      dst[row * DIM + col] = src[row * PACKED_OUTPUT_COLS + col * 4];
    }
  }
}

static void hw_matmul(elem_t *a, elem_t *b, result_t *c, int size,
                      int ws_mode) {
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  uint32_t acc_bank_id = 2;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);
  bb_mem_alloc(acc_bank_id, 1, 4);

  pack_input_matrix(a, packed_input_matrix_a);
  pack_input_matrix(b, packed_input_matrix_b);

  bb_mvin((uintptr_t)packed_input_matrix_a, op1_bank_id, DIM, 1);
  bb_mvin((uintptr_t)packed_input_matrix_b, op2_bank_id, DIM, 1);
  if (ws_mode) {
    bb_BFP_WS(op1_bank_id, op2_bank_id, acc_bank_id, size);
  } else {
    bb_BFP_OS(op1_bank_id, op2_bank_id, acc_bank_id, size);
  }
  bb_mvout((uintptr_t)packed_output_matrix, acc_bank_id, size, 1);
  bb_fence();
  unpack_output_matrix(packed_output_matrix, c);
}

static int run_test(const char *test_name, int ws_mode) {
  clear_u32_matrix(output_matrix, DIM, DIM);
  clear_u32_matrix(expected_matrix, DIM, DIM);
  clear_u32_matrix(packed_output_matrix, DIM, PACKED_OUTPUT_COLS);
  cpu_matmul(input_matrix_a, input_matrix_b, expected_matrix, DIM, DIM, DIM);
  hw_matmul(input_matrix_a, input_matrix_b, output_matrix, DIM, ws_mode);

  if (!compare_u32_matrices(output_matrix, expected_matrix, DIM, DIM)) {
    printf("%s FAILED\n", test_name);
    return 0;
  }

  printf("%s PASSED\n", test_name);
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  init_u8_random_matrix(input_matrix_a, DIM, DIM, 123);
  init_u8_random_matrix(input_matrix_b, DIM, DIM, 456);

  int os_passed = run_test("systolic_os_random", 0);
  int ws_passed = run_test("systolic_ws_random", 1);
  return (os_passed && ws_passed) ? 0 : 1;

#ifdef MULTICORE
  exit(0);
#endif
}
