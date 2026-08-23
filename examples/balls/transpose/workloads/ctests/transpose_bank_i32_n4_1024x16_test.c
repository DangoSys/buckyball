#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/transpose.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ROWS BANK_LINES
#define N_GROUPS 4
#define COLS (N_GROUPS * ((BANK_WIDTH) / 32))
#define ELEM_BITS 32
#define SEED 0x54

static int32_t input_matrix[ROWS * COLS] __attribute__((aligned(64)));
static int32_t output_matrix[COLS * ROWS] __attribute__((aligned(64)));
static int32_t expected_matrix[COLS * ROWS] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  init_i32_random_matrix(input_matrix, ROWS, COLS, SEED);
  for (int r = 0; r < ROWS; ++r) {
    for (int c = 0; c < COLS; ++c) {
      expected_matrix[c * ROWS + r] = input_matrix[r * COLS + c];
    }
  }
  memset(output_matrix, 0, sizeof(output_matrix));

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

  for (int i = 0; i < COLS * ROWS; ++i) {
    if (output_matrix[i] != expected_matrix[i]) {
      printf("Transpose bank i32 n4 1024x16 FAILED at %d\n", i);
      return 1;
    }
  }
  printf("Transpose bank i32 n4 1024x16 PASSED\n");
  return 0;
}
