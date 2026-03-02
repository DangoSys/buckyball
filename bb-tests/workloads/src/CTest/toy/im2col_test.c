#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16
#define KROW 4
#define KCOL 1
#define INROW 16
#define INCOL 16
#define STARTROW 0
#define STARTCOL 0

#define EXPECTED_ROWS 1024
#define EXPECTED_COLS 4
#define LANES_PER_BEAT 16

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[EXPECTED_ROWS * EXPECTED_COLS]
    __attribute__((aligned(64)));
static elem_t expected_matrix[EXPECTED_ROWS * EXPECTED_COLS]
    __attribute__((aligned(64)));

static int conv_num(void) {
  return (INROW - KROW + 1 - STARTROW) * (INCOL - KCOL + 1 - STARTCOL);
}

static int kernel_elems(void) { return KROW * KCOL; }

static void build_expected_im2col_matrix(elem_t *input, elem_t *expected,
                                         int inrow, int incol, int krow,
                                         int kcol, int startrow, int startcol) {
  clear_i8_matrix(expected, EXPECTED_ROWS, EXPECTED_COLS);

  int row_end = inrow - krow;
  int col_end = incol - kcol;
  int kernel = krow * kcol;
  int window_idx = 0;

  for (int r = startrow; r <= row_end; r++) {
    for (int c = startcol; c <= col_end; c++) {
      int elem_idx = 0;
      for (int kr = 0; kr < krow; kr++) {
        for (int kc = 0; kc < kcol; kc++) {
          expected[window_idx * kernel + elem_idx] =
              input[(r + kr) * incol + (c + kc)];
          elem_idx++;
        }
      }
      window_idx++;
    }
  }
}

void hw_im2col(const char *test_name, elem_t *a, elem_t *b, int size) {
  (void)test_name;
  (void)size;

  // spad0: operand A, offset 0
  uint32_t op1_bank_id = 0;
  // spad1: operand B, offset 0
  uint32_t op2_bank_id = 1;
  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);

  bb_mvin((uintptr_t)a, op1_bank_id, 32, 1);
  uint64_t krow = KROW;
  uint64_t kcol = KCOL;
  uint64_t inrow = INROW;
  uint64_t incol = INCOL;
  uint64_t startrow = STARTROW;
  uint64_t startcol = STARTCOL;
  // bb_im2col(op1_bank_id, op2_bank_id, krow, kcol, inrow, incol, startrow,
  // startcol);
  bb_im2col(op1_bank_id, op2_bank_id, krow, kcol, inrow, incol, startrow,
            startcol);
  bb_mvout((uintptr_t)b, op2_bank_id, conv_num() / kernel_elems(), 1);
  bb_fence();
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  int conv = conv_num();
  int elems = kernel_elems();

  clear_i8_matrix(b, EXPECTED_ROWS, EXPECTED_COLS);
  build_expected_im2col_matrix(a, expected_matrix, INROW, INCOL, KROW, KCOL,
                               STARTROW, STARTCOL);

  hw_im2col(test_name, a, b, size);

  if (compare_i8_matrices(b, expected_matrix, conv, elems)) {
    printf("%s compare test PASSED\n", test_name);
    return 1;
  } else {
    printf("%s compare test FAILED\n", test_name);
    return 0;
  }
}

int test_im2col() {
  init_sequence_matrix(input_matrix_a, DIM, DIM);
  return run_test("Im2col", input_matrix_a, output_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_im2col();
  if (passed) {
    printf("Im2col test PASSED\n");
  } else {
    printf("Im2col test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
