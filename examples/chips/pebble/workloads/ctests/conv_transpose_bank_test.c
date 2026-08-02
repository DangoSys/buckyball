#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <bbhw/mem/params.h>
#include <stdio.h>

#define ROWS BANK_LINES
#define COLS 16
#define ELEM_BITS 8
#define SEED 0xC2

static elem_t input_matrix[ROWS * COLS] __attribute__((aligned(64)));
static elem_t output_matrix[COLS * ROWS] __attribute__((aligned(64)));
static elem_t expected_matrix[COLS * ROWS] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  init_i8_random_matrix(input_matrix, ROWS, COLS, SEED);
  transpose_u8_matrix(input_matrix, expected_matrix, ROWS, COLS);
  bb_mem_alloc(0, BANK_LINES, 1);
  bb_mem_alloc(1, BANK_LINES, 1);
  bb_mvin((uintptr_t)input_matrix, 0, ROWS, 1);
  bb_transpose(0, 1, ROWS, ELEM_BITS);
  bb_mvout((uintptr_t)output_matrix, 1, ROWS, 1);
  bb_fence();
  int ok = compare_i8_matrices(output_matrix, expected_matrix, COLS, ROWS);
  printf("conv_transpose_bank %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
