#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

enum {
  INPUT_H = 6,
  INPUT_W = 6,
  KERNEL_H = 3,
  KERNEL_W = 3,
  STRIDE = 1,
  PADDING = 0,
  DILATION = 1,
  OUTPUT_H = 4,
  OUTPUT_W = 4,
  NUM_WINDOWS = OUTPUT_H * OUTPUT_W,
  KERNEL_ELEMS = KERNEL_H * KERNEL_W,
  LANES_PER_BANK_ROW = 16,
  INPUT_BEATS =
      (INPUT_H * INPUT_W + LANES_PER_BANK_ROW - 1) / LANES_PER_BANK_ROW,
  TRANSPOSE_SOURCE_ROWS = LANES_PER_BANK_ROW,
  TRANSPOSE_SOURCE_ELEMS = TRANSPOSE_SOURCE_ROWS * KERNEL_ELEMS,
  ACC_LANES_PER_ROW = 16,
  PACKED_C_ELEMS = NUM_WINDOWS * ACC_LANES_PER_ROW,
};

static const elem_t input[INPUT_H * INPUT_W] __attribute__((aligned(64))) = {
    1, 2, 0,  3, -1, 2,  4, -2, 1, 0, 2,  3, 0,  1, 3, -1, 4, 2,
    2, 3, -2, 1, 0,  -1, 1, 0,  2, 4, -3, 2, -1, 2, 1, 0,  3, 1,
};

static const elem_t kernel_flat[KERNEL_ELEMS] = {
    1, -2, 3, 0, 2, -1, 4, 1, -3,
};

/*
 * Logical 16x9 Im2Col result. The hardware stores each logical row in one
 * 16-byte bank row, with lanes 9..15 zero-filled for MatrixBall.
 */
static const elem_t expected_a[NUM_WINDOWS * KERNEL_ELEMS]
    __attribute__((unused)) = {
        1,  2,  0,  4,  -2, 1,  0,  1,  3,  2,  0,  3,  -2, 1,  0,  1,  3,  -1,
        0,  3,  -1, 1,  0,  2,  3,  -1, 4,  3,  -1, 2,  0,  2,  3,  -1, 4,  2,

        4,  -2, 1,  0,  1,  3,  2,  3,  -2, -2, 1,  0,  1,  3,  -1, 3,  -2, 1,
        1,  0,  2,  3,  -1, 4,  -2, 1,  0,  0,  2,  3,  -1, 4,  2,  1,  0,  -1,

        0,  1,  3,  2,  3,  -2, 1,  0,  2,  1,  3,  -1, 3,  -2, 1,  0,  2,  4,
        3,  -1, 4,  -2, 1,  0,  2,  4,  -3, -1, 4,  2,  1,  0,  -1, 4,  -3, 2,

        2,  3,  -2, 1,  0,  2,  -1, 2,  1,  3,  -2, 1,  0,  2,  4,  2,  1,  0,
        -2, 1,  0,  2,  4,  -3, 1,  0,  3,  1,  0,  -1, 4,  -3, 2,  0,  3,  1,
};

static const result_t expected_c[NUM_WINDOWS] = {
    -16, 23, -12, 6, 27, 10, -6, 18, 13, -23, 40, 5, -17, 19, 2, -10,
};

/*
 * TransposeBall consumes a physical 16xKERNEL_ELEMS matrix. Row zero is the
 * logical 1x9 kernel_flat and rows 1..15 are padding. The transposed B bank
 * consequently contains a logical 9x1 matrix in lane zero of rows 0..8.
 */
static elem_t transpose_source[TRANSPOSE_SOURCE_ELEMS]
    __attribute__((aligned(64)));

/*
 * A 4-bank accumulator exposes 16 int32 lanes per physical row. MatrixBall is
 * invoked with N=1, so only lane zero of each of the 16 rows is logical C.
 */
static result_t packed_c[PACKED_C_ELEMS] __attribute__((aligned(64)));

static void initialize_buffers(void) {
  for (int i = 0; i < TRANSPOSE_SOURCE_ELEMS; ++i) {
    transpose_source[i] = 0;
  }
  for (int i = 0; i < KERNEL_ELEMS; ++i) {
    transpose_source[i] = kernel_flat[i];
  }

  for (int i = 0; i < PACKED_C_ELEMS; ++i) {
    packed_c[i] = (result_t)0x5a5a5a5a;
  }
}

static void run_hardware_pipeline(void) {
  const uint32_t input_bank_id = 0;
  const uint32_t matrix_a_bank_id = 1;
  const uint32_t kernel_source_bank_id = 2;
  const uint32_t matrix_b_bank_id = 3;
  const uint32_t matrix_c_bank_id = 4;

  bb_mem_alloc(input_bank_id, 1, 1);
  bb_mem_alloc(matrix_a_bank_id, 1, 1);
  bb_mem_alloc(kernel_source_bank_id, 1, 1);
  bb_mem_alloc(matrix_b_bank_id, 1, 1);
  bb_mem_alloc(matrix_c_bank_id, 1, 4);

  bb_mvin((uintptr_t)input, input_bank_id, INPUT_BEATS, 1);
  bb_im2col(input_bank_id, matrix_a_bank_id, INPUT_H, KERNEL_H, STRIDE,
            PADDING);
  bb_fence();

  bb_mvin((uintptr_t)transpose_source, kernel_source_bank_id, KERNEL_ELEMS, 1);
  bb_transpose(kernel_source_bank_id, matrix_b_bank_id, KERNEL_ELEMS, 0);
  bb_fence();

  bb_matrix_mnk(matrix_a_bank_id, matrix_b_bank_id, matrix_c_bank_id,
                NUM_WINDOWS, 1, KERNEL_ELEMS);
  bb_mvout((uintptr_t)packed_c, matrix_c_bank_id, NUM_WINDOWS, 1);
  bb_fence();
}

static int compare_output(void) {
  int passed = 1;

  for (int i = 0; i < NUM_WINDOWS; ++i) {
    const result_t actual = packed_c[i * ACC_LANES_PER_ROW];
    if (actual != expected_c[i]) {
      printf("Mismatch at C[%d] / output[%d][%d]: expected=%d, actual=%d\n", i,
             i / OUTPUT_W, i % OUTPUT_W, (int)expected_c[i], (int)actual);
      passed = 0;
    }
  }

  return passed;
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  initialize_buffers();
  run_hardware_pipeline();

  const int passed = compare_output();
  printf("conv_im2col_transpose_matmul_6x6_test: %s\n",
         passed ? "PASS" : "FAIL");

#ifdef MULTICORE
  exit(passed ? 0 : 1);
#endif
  return passed ? 0 : 1;
}
