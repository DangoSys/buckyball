#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/transpose.h>
#include <stdio.h>

#define ROWS BANK_LINES
#define N_GROUPS 1
#define COLS (N_GROUPS * ((BANK_WIDTH) / 8))
#define ELEM_BITS 8
#define SEED 0x51

static elem_t input_matrix[ROWS * COLS] __attribute__((aligned(64)));
static elem_t output_matrix[COLS * ROWS] __attribute__((aligned(64)));
static elem_t expected_matrix[COLS * ROWS] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  init_i8_random_matrix(input_matrix, ROWS, COLS, SEED);
  transpose_u8_matrix(input_matrix, expected_matrix, ROWS, COLS);
  clear_i8_matrix(output_matrix, COLS, ROWS);

  const uint32_t src = 0;
  const uint32_t dst = 1;

  bb_mem_alloc(src, BANK_LINES, N_GROUPS);
  bb_mem_alloc(dst, BANK_LINES, N_GROUPS);
  bb_mvin((uintptr_t)input_matrix, src, ROWS, 1);
  bb_transpose(src, dst, ROWS, ELEM_BITS);
  bb_mvout((uintptr_t)output_matrix, dst, ROWS, 1);
  bb_fence();
  bb_mem_release(src);
  bb_mem_release(dst);

  if (compare_i8_matrices(output_matrix, expected_matrix, COLS, ROWS)) {
    printf("Transpose bank i8 n1 1024x16 PASSED\n");
    return 0;
  }
  printf("Transpose bank i8 n1 1024x16 FAILED\n");
  return 1;
}
