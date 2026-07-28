#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 16
#define K_SIZE 5
#define STRIDE 1
#define PADDING 0

enum {
  LANES_PER_BEAT = 16,
  OUTPUT_DIM = (DIM + 2 * PADDING - K_SIZE) / STRIDE + 1,
  NUM_WINDOWS = OUTPUT_DIM * OUTPUT_DIM,
  KERNEL_ELEMS = K_SIZE * K_SIZE,
  INPUT_ELEMS = DIM * 64,
  OUTPUT_ELEMS = NUM_WINDOWS * KERNEL_ELEMS,
  INPUT_BEATS = (INPUT_ELEMS + LANES_PER_BEAT - 1) / LANES_PER_BEAT,
  OUTPUT_BEATS = (OUTPUT_ELEMS + LANES_PER_BEAT - 1) / LANES_PER_BEAT,
};

static elem_t input_matrix[INPUT_ELEMS] __attribute__((aligned(64)));
static elem_t output_matrix[OUTPUT_ELEMS] __attribute__((aligned(64)));
static elem_t expected_matrix[OUTPUT_ELEMS] __attribute__((aligned(64)));

static void build_expected_im2col(void) {
  int output_index = 0;

  for (int output_row = 0; output_row < OUTPUT_DIM; output_row++) {
    for (int output_col = 0; output_col < OUTPUT_DIM; output_col++) {
      int input_row = output_row * STRIDE - PADDING;
      int input_col = output_col * STRIDE - PADDING;

      for (int kernel_row = 0; kernel_row < K_SIZE; kernel_row++) {
        for (int kernel_col = 0; kernel_col < K_SIZE; kernel_col++) {
          int row = input_row + kernel_row;
          int col = input_col + kernel_col;

          if (row < 0 || row >= DIM || col < 0 || col >= DIM) {
            expected_matrix[output_index++] = 0;
          } else {
            expected_matrix[output_index++] = input_matrix[row * DIM + col];
          }
        }
      }
    }
  }
}

static void hw_im2col(elem_t *input, elem_t *output) {
  uint32_t input_bank_id = 0;
  uint32_t output_bank_id = 1;

  bb_mem_alloc(input_bank_id, 1, 1);
  bb_mem_alloc(output_bank_id, 1, 1);

  bb_mvin((uintptr_t)input, input_bank_id, INPUT_BEATS, 1);
  bb_im2col(input_bank_id, output_bank_id, DIM, K_SIZE, STRIDE, PADDING);
  bb_mvout((uintptr_t)output, output_bank_id, OUTPUT_BEATS, 1);
  bb_fence();
}

static int run_test(const char *test_name) {
  // Initialize input: A[i][j] = i + j
  init_sequence_matrix(input_matrix, DIM, DIM);
  clear_i8_matrix(output_matrix, NUM_WINDOWS, KERNEL_ELEMS);

  // Compute expected im2col output on CPU
  build_expected_im2col();

  // Run hardware im2col
  hw_im2col(input_matrix, output_matrix);

  // Compare each flattened K_SIZE x K_SIZE window
  if (compare_i8_matrices(output_matrix, expected_matrix, NUM_WINDOWS,
                          KERNEL_ELEMS)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = run_test("im2col 16x16 kernel 5x5");

  if (passed) {
    printf("All im2col tests PASSED\n");
  } else {
    printf("Some im2col tests FAILED\n");
  }

#ifdef MULTICORE
  exit(passed ? 0 : 1);
#endif
  return passed ? 0 : 1;
}
