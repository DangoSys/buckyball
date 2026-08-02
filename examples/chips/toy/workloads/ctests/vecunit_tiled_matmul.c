#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <string.h>

#define DIM 16
#define KDIM 64
#define KTILE 32

_Static_assert(KDIM % KTILE == 0, "KDIM % KTILE");
_Static_assert(KTILE % 16 == 0, "KTILE % 16");

static elem_t input_a[DIM * KDIM] __attribute__((aligned(64)));
static elem_t input_b[KDIM * DIM] __attribute__((aligned(64)));
static elem_t tile_a[DIM * KTILE] __attribute__((aligned(64)));
static result_t output[DIM * DIM] __attribute__((aligned(64)));
static result_t expected[DIM * DIM] __attribute__((aligned(64)));
static result_t zero[DIM * DIM] __attribute__((aligned(64))) = {0};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  clear_u8_matrix(input_a, DIM, KDIM);
  clear_u8_matrix(input_b, KDIM, DIM);
  clear_u32_matrix(expected, DIM, DIM);
  for (int k = 0; k < KDIM; k++) {
    int i = k % DIM;
    input_a[i * KDIM + k] = 1;
    input_b[k * DIM + i] = 1;
  }
  int diag = KDIM / DIM;
  for (int r = 0; r < DIM; r++) {
    expected[r * DIM + r] = diag;
  }

  const uint32_t op1 = 0;
  const uint32_t op2 = 1;
  const uint32_t acc = 2;
  const uint32_t a_t = 3;
  bb_mem_alloc(op1, 1, 1);
  bb_mem_alloc(op2, 1, 1);
  bb_mem_alloc(acc, 1, 4);
  bb_mem_alloc(a_t, 1, 1);
  bb_mvin((uintptr_t)zero, acc, DIM, 1);

  for (int k0 = 0; k0 < KDIM; k0 += KTILE) {
    for (int r = 0; r < DIM; r++) {
      memcpy(&tile_a[r * KTILE], &input_a[r * KDIM + k0], (size_t)KTILE);
    }
    bb_mvin((uintptr_t)tile_a, op1, DIM * (KTILE / 16), 1);
    bb_mvin((uintptr_t)(input_b + k0 * DIM), op2, KTILE, 1);
    bb_transpose(op1, a_t, KTILE, 8);
    bb_mul_warp16(a_t, op2, acc, KTILE, 0);
  }

  bb_mvout((uintptr_t)output, acc, DIM, 1);
  bb_fence();

  int passed = compare_u32_matrices(output, expected, DIM, DIM);
  printf("tiled_matmul %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
