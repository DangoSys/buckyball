#include "pebble.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Change only this name to select an input/kernel/golden-answer group from
 * pebble/workloads/common/pebble.c.
 */
#ifndef TEST_CASE_NAME
#define TEST_CASE_NAME "conv_17x17_k7"
#endif

enum {
  MAX_INPUT_ELEMS = 17 * 17,
  MAX_KERNEL_ELEMS = 7 * 7,
  MAX_WINDOWS = 11 * 11,
  MAX_M_TILES = (MAX_WINDOWS + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES,
  MAX_K_TILES = (MAX_KERNEL_ELEMS + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES,
  MAX_IM2COL_ROWS = MAX_M_TILES * MAX_K_TILES * PEBBLE_INT8_LANES,
  MAX_IM2COL_ELEMS = MAX_IM2COL_ROWS * PEBBLE_INT8_LANES,
  MAX_TRANSPOSE_SOURCE_ELEMS = PEBBLE_INT8_LANES * MAX_KERNEL_ELEMS,
  MAX_TRANSPOSE_ELEMS = MAX_KERNEL_ELEMS * PEBBLE_INT8_LANES,
  MAX_PACKED_C_ELEMS = MAX_WINDOWS * PEBBLE_ACC_LANES,
};

static elem_t actual_im2col[MAX_IM2COL_ELEMS] __attribute__((aligned(64)));
static elem_t logical_im2col[MAX_WINDOWS * MAX_KERNEL_ELEMS]
    __attribute__((aligned(64)));
static elem_t packed_input[(MAX_INPUT_ELEMS + PEBBLE_INT8_LANES - 1) /
                           PEBBLE_INT8_LANES * PEBBLE_INT8_LANES]
    __attribute__((aligned(64)));
static elem_t transpose_source[MAX_TRANSPOSE_SOURCE_ELEMS]
    __attribute__((aligned(64)));
static elem_t actual_transpose[MAX_TRANSPOSE_ELEMS]
    __attribute__((aligned(64)));
static result_t packed_c[MAX_PACKED_C_ELEMS] __attribute__((aligned(64)));
static result_t logical_c[MAX_WINDOWS] __attribute__((aligned(64)));

static int im2col_physical_index(int logical_row, int logical_col,
                                 int k_tiles) {
  const int m_tile = logical_row / PEBBLE_INT8_LANES;
  const int m_row = logical_row % PEBBLE_INT8_LANES;
  const int k_tile = logical_col / PEBBLE_INT8_LANES;
  const int lane = logical_col % PEBBLE_INT8_LANES;
  const int bank_row = (m_tile * k_tiles + k_tile) * PEBBLE_INT8_LANES + m_row;
  return bank_row * PEBBLE_INT8_LANES + lane;
}

static void initialize_buffers(const pebble_conv_test_case_t *test) {
  const int input_elems = test->input_h * test->input_w;
  const int kernel_elems = test->kernel_h * test->kernel_w;

  for (int i = 0; i < (int)(sizeof(packed_input) / sizeof(packed_input[0]));
       ++i) {
    packed_input[i] = 0;
  }
  for (int i = 0; i < input_elems; ++i) {
    packed_input[i] = test->input[i];
  }
  for (int i = 0; i < MAX_IM2COL_ELEMS; ++i) {
    actual_im2col[i] = (elem_t)0x5a;
  }
  for (int i = 0; i < MAX_TRANSPOSE_SOURCE_ELEMS; ++i) {
    transpose_source[i] = 0;
  }
  for (int i = 0; i < kernel_elems; ++i) {
    transpose_source[i] = test->kernel[i];
  }
  for (int i = 0; i < MAX_TRANSPOSE_ELEMS; ++i) {
    actual_transpose[i] = (elem_t)0x5a;
  }
  for (int i = 0; i < MAX_PACKED_C_ELEMS; ++i) {
    packed_c[i] = (result_t)0x5a5a5a5a;
  }
}

static int check_and_print_im2col(const pebble_conv_test_case_t *test,
                                  int k_tiles) {
  const int windows = test->output_h * test->output_w;
  const int kernel_elems = test->kernel_h * test->kernel_w;
  const int m_tiles = (windows + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;

  for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
      for (int m_row = 0; m_row < PEBBLE_INT8_LANES; ++m_row) {
        for (int lane = 0; lane < PEBBLE_INT8_LANES; ++lane) {
          const int row = m_tile * PEBBLE_INT8_LANES + m_row;
          const int col = k_tile * PEBBLE_INT8_LANES + lane;
          const int physical_index = im2col_physical_index(row, col, k_tiles);
          const elem_t actual = actual_im2col[physical_index];
          const elem_t expected =
              (row < windows && col < kernel_elems)
                  ? test->expected_im2col[row * kernel_elems + col]
                  : 0;
          if (actual != expected) {
            printf("FAIL stage=Im2ColBall row=%d col=%d expected=%d "
                   "actual=%d\n",
                   row, col, (int)expected, (int)actual);
            return 0;
          }
          if (row < windows && col < kernel_elems) {
            logical_im2col[row * kernel_elems + col] = actual;
          }
        }
      }
    }
  }

  printf("PASS stage=Im2ColBall\n");
  pebble_print_i8_matrix("Im2ColBall output", logical_im2col, windows,
                         kernel_elems);
  return 1;
}

static int check_and_print_transpose(const pebble_conv_test_case_t *test) {
  const int kernel_elems = test->kernel_h * test->kernel_w;

  for (int row = 0; row < kernel_elems; ++row) {
    for (int col = 0; col < PEBBLE_INT8_LANES; ++col) {
      const elem_t expected = (col == 0) ? test->kernel[row] : 0;
      const elem_t actual = actual_transpose[row * PEBBLE_INT8_LANES + col];
      if (actual != expected) {
        printf("FAIL stage=TransposeBall row=%d col=%d expected=%d actual=%d\n",
               row, col, (int)expected, (int)actual);
        return 0;
      }
    }
  }

  printf("PASS stage=TransposeBall\n");
  pebble_print_i8_matrix("TransposeBall output", actual_transpose, kernel_elems,
                         PEBBLE_INT8_LANES);
  return 1;
}

static int check_and_print_matrix(const pebble_conv_test_case_t *test) {
  const int windows = test->output_h * test->output_w;

  for (int i = 0; i < windows; ++i) {
    const result_t actual = packed_c[i * PEBBLE_ACC_LANES];
    const result_t expected = test->expected_output[i];
    if (actual != expected) {
      printf("FAIL stage=MatrixBall row=%d col=%d expected=%d actual=%d\n",
             i / test->output_w, i % test->output_w, (int)expected,
             (int)actual);
      return 0;
    }
    logical_c[i] = actual;
  }

  printf("PASS stage=MatrixBall\n");
  pebble_print_i32_matrix("MatrixBall output", logical_c, test->output_h,
                          test->output_w);
  return 1;
}

static int run_hardware_pipeline(const pebble_conv_test_case_t *test) {
  const int input_elems = test->input_h * test->input_w;
  const int kernel_elems = test->kernel_h * test->kernel_w;
  const int windows = test->output_h * test->output_w;
  const int input_beats =
      (input_elems + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;
  const int m_tiles = (windows + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;
  const int k_tiles =
      (kernel_elems + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;
  const int im2col_rows = m_tiles * k_tiles * PEBBLE_INT8_LANES;

  const uint32_t input_bank = 0;
  const uint32_t matrix_a_bank = 1;
  const uint32_t kernel_source_bank = 2;
  const uint32_t matrix_b_bank = 3;
  const uint32_t matrix_c_bank = 4;

  bb_mem_alloc(input_bank, 1, 1);
  bb_mem_alloc(matrix_a_bank, 1, 1);
  bb_mvin((uintptr_t)packed_input, input_bank, input_beats, 1);
  bb_im2col(input_bank, matrix_a_bank, test->input_h, test->kernel_h,
            test->stride, test->padding);
  bb_mvout((uintptr_t)actual_im2col, matrix_a_bank, im2col_rows, 1);
  bb_fence();
  if (!check_and_print_im2col(test, k_tiles)) {
    return 0;
  }
  bb_mem_release(input_bank);

  bb_mem_alloc(kernel_source_bank, 1, 1);
  bb_mem_alloc(matrix_b_bank, 1, 1);
  bb_mvin((uintptr_t)transpose_source, kernel_source_bank, kernel_elems, 1);
  bb_transpose(kernel_source_bank, matrix_b_bank, kernel_elems, 0);
  bb_mvout((uintptr_t)actual_transpose, matrix_b_bank, kernel_elems, 1);
  bb_fence();
  if (!check_and_print_transpose(test)) {
    return 0;
  }
  bb_mem_release(kernel_source_bank);

  bb_mem_alloc(matrix_c_bank, 1, 4);
  bb_matrix_mnk(matrix_a_bank, matrix_b_bank, matrix_c_bank, windows, 1,
                kernel_elems);
  bb_mvout((uintptr_t)packed_c, matrix_c_bank, windows, 1);
  bb_fence();
  if (!check_and_print_matrix(test)) {
    return 0;
  }

  bb_mem_release(matrix_a_bank);
  bb_mem_release(matrix_b_bank);
  bb_mem_release(matrix_c_bank);
  return 1;
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  const pebble_conv_test_case_t *test =
      pebble_find_conv_test_case(TEST_CASE_NAME);
  if (test == NULL) {
    printf("Unknown TEST_CASE_NAME: %s\n", TEST_CASE_NAME);
#ifdef MULTICORE
    exit(1);
#endif
    return 1;
  }

  printf("Running Pebble test case: %s\n", test->name);
  initialize_buffers(test);
  const int passed = run_hardware_pipeline(test);
  printf("conv_im2col_transpose_matmul_test [%s]: %s\n", test->name,
         passed ? "PASS" : "FAIL");

#ifdef MULTICORE
  exit(passed ? 0 : 1);
#endif
  return passed ? 0 : 1;
}
