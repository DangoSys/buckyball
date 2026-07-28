#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 6
#define K_SIZE 3
#define STRIDE 1
#define PADDING 0

enum {
  LANES_PER_BEAT = 16,
  OUTPUT_DIM = (DIM + 2 * PADDING - K_SIZE) / STRIDE + 1,
  NUM_WINDOWS = OUTPUT_DIM * OUTPUT_DIM,
  KERNEL_ELEMS = K_SIZE * K_SIZE,
  M_TILES = (NUM_WINDOWS + LANES_PER_BEAT - 1) / LANES_PER_BEAT,
  K_TILES = (KERNEL_ELEMS + LANES_PER_BEAT - 1) / LANES_PER_BEAT,
  INPUT_ELEMS = DIM * DIM,
  OUTPUT_ROWS = M_TILES * K_TILES * LANES_PER_BEAT,
  OUTPUT_ELEMS = OUTPUT_ROWS * LANES_PER_BEAT,
  INPUT_BEATS = (INPUT_ELEMS + LANES_PER_BEAT - 1) / LANES_PER_BEAT,
};

static elem_t input_matrix[INPUT_ELEMS] __attribute__((aligned(64)));
static elem_t output_matrix[OUTPUT_ELEMS] __attribute__((aligned(64)));
static elem_t expected_matrix[OUTPUT_ELEMS] __attribute__((aligned(64)));

static void build_expected_im2col(void) {
  clear_i8_matrix(expected_matrix, OUTPUT_ROWS, LANES_PER_BEAT);
  int window_index = 0;

  for (int output_row = 0; output_row < OUTPUT_DIM; output_row++) {
    for (int output_col = 0; output_col < OUTPUT_DIM; output_col++) {
      int input_row = output_row * STRIDE - PADDING;
      int input_col = output_col * STRIDE - PADDING;

      for (int kernel_row = 0; kernel_row < K_SIZE; kernel_row++) {
        for (int kernel_col = 0; kernel_col < K_SIZE; kernel_col++) {
          int kernel_index = kernel_row * K_SIZE + kernel_col;
          int m_tile = window_index / LANES_PER_BEAT;
          int m_row = window_index % LANES_PER_BEAT;
          int k_tile = kernel_index / LANES_PER_BEAT;
          int lane = kernel_index % LANES_PER_BEAT;
          int bank_row = (m_tile * K_TILES + k_tile) * LANES_PER_BEAT + m_row;
          int output_index = bank_row * LANES_PER_BEAT + lane;
          int row = input_row + kernel_row;
          int col = input_col + kernel_col;

          if (row < 0 || row >= DIM || col < 0 || col >= DIM) {
            expected_matrix[output_index] = 0;
          } else {
            expected_matrix[output_index] = input_matrix[row * DIM + col];
          }
        }
      }
      window_index++;
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
  bb_mvout((uintptr_t)output, output_bank_id, OUTPUT_ROWS, 1);
  bb_fence();
}

static int run_test(const char *test_name) {
  // Initialize input: A[i][j] = i + j
  init_sequence_matrix(input_matrix, DIM, DIM);
  clear_i8_matrix(output_matrix, OUTPUT_ROWS, LANES_PER_BEAT);

  // Compute expected im2col output on CPU
  build_expected_im2col();

  // Run hardware im2col
  hw_im2col(input_matrix, output_matrix);

  // Compare MatrixBall-compatible (M tile, K tile, tile row, lane) layout.
  if (compare_i8_matrices(output_matrix, expected_matrix, OUTPUT_ROWS,
                          LANES_PER_BEAT)) {
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

  int passed = run_test("im2col 6x6 kernel 3x3 stride 1 padding 0");

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
