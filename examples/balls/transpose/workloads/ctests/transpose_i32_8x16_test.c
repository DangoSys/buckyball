#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/transpose.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ROWS 8
#define N_GROUPS 4
#define COLS (N_GROUPS * ((BANK_WIDTH) / 32))
#define ELEM_BITS 32

static int32_t input_matrix[ROWS * COLS] __attribute__((aligned(64))) = {
    10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
    25,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,
    63,  64,  65,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
    82,  83,  84,  85,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100,
    101, 102, 103, 104, 105, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 130, 131, 132, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 150, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165,
};

static int32_t output_matrix[COLS * ROWS] __attribute__((aligned(64)));
static int32_t expected_matrix[COLS * ROWS] __attribute__((aligned(64)));

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  for (int r = 0; r < ROWS; ++r) {
    for (int c = 0; c < COLS; ++c) {
      expected_matrix[c * ROWS + r] = input_matrix[r * COLS + c];
    }
  }
  memset(output_matrix, 0, sizeof(output_matrix));

  const uint32_t src = 0;
  const uint32_t dst = 1;

  bb_mem_alloc(src, 1, N_GROUPS);
  bb_mem_alloc(dst, 1, N_GROUPS);
  bb_mvin((uintptr_t)input_matrix, src, ROWS, 1);
  bb_transpose(src, dst, ROWS, ELEM_BITS);
  bb_mvout((uintptr_t)output_matrix, dst, ROWS, 1);
  bb_fence();
  bb_mem_release(src);
  bb_mem_release(dst);

  for (int i = 0; i < COLS * ROWS; ++i) {
    if (output_matrix[i] != expected_matrix[i]) {
      printf("Transpose i32 8x16 FAILED at %d\n", i);
      return 1;
    }
  }
  printf("Transpose i32 8x16 PASSED\n");
  return 0;
}
