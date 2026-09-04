#ifndef PEBBLE_H
#define PEBBLE_H

#include "buckyball.h"
#include <params.h>
#include <stddef.h>

enum {
  PEBBLE_INT8_LANES = BANK_WIDTH / 8,
  PEBBLE_ACC_LANES = BANK_WIDTH / 8,
};

/*
 * A convolution test case contains all input data and golden answers needed by
 * the Im2Col -> Transpose -> Matrix pipeline.  Keeping these answers in common
 * avoids generating them on the simulated CPU.
 *
 * To add a test case:
 *   1. Define const elem_t input[], kernel[], and expected_im2col[] arrays plus
 *      a const result_t expected_output[] array in pebble.c.
 *   2. expected_im2col is the logical row-major matrix
 *      [number_of_windows][kernel_h * kernel_w], not the bank-padded layout.
 *   3. expected_output is the logical row-major output matrix
 *      [output_h][output_w].
 *   4. Add one entry to convolution_test_cases[] in pebble.c.
 *   5. Select it in a ctest by changing only TEST_CASE_NAME.
 *
 * The transpose golden answer is kernel itself: TransposeBall turns the
 * physical 16xK source (kernel in row 0, remaining rows zero) into Kx16, where
 * column 0 is kernel and columns 1..15 are zero.
 */
typedef struct {
  const char *name;
  int input_h;
  int input_w;
  int kernel_h;
  int kernel_w;
  int stride;
  int padding;
  int dilation;
  int output_h;
  int output_w;
  const elem_t *input;
  const elem_t *kernel;
  const elem_t *expected_im2col;
  const result_t *expected_output;
} pebble_conv_test_case_t;

const pebble_conv_test_case_t *pebble_find_conv_test_case(const char *name);

void pebble_print_i8_matrix(const char *name, const elem_t *matrix, int rows,
                            int cols);
void pebble_print_i32_matrix(const char *name, const result_t *matrix, int rows,
                             int cols);

#endif
