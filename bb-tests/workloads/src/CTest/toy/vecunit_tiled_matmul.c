#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 16
#define KDIM 1024
#define KTILE 512

_Static_assert(KDIM % KTILE == 0, "KDIM must be divisible by KTILE");
_Static_assert(KDIM % DIM == 0, "KDIM must be divisible by DIM");
_Static_assert(KTILE % 16 == 0,
               "KTILE must be multiple of 16 (mvin line size)");

static elem_t input_matrix_a[DIM * KDIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[KDIM * DIM] __attribute__((aligned(64)));
static elem_t tile_a[DIM * KTILE] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t zero_matrix[DIM * DIM] __attribute__((aligned(64))) = {0};

/// C = A * B with A row-major DIMxKDIM and B row-major KDIMxDIM; K is tiled by
/// KTILE and accumulated into acc.
void hw_matmul_tiled(const char *test_name, elem_t *a, elem_t *b, result_t *c) {
  (void)test_name;
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  int acc_bank_id = 2;
  uint32_t a_transposed_bank_id = 3;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);
  bb_mem_alloc(acc_bank_id, 1, 4);
  bb_mem_alloc(a_transposed_bank_id, 1, 1);

  bb_mvin((uintptr_t)zero_matrix, acc_bank_id, DIM, 1);

  for (int k0 = 0; k0 < KDIM; k0 += KTILE) {
    /* Row-major A: column block [k0, k0+KTILE) is not contiguous across rows;
     * pack to tile_a. */
    for (int r = 0; r < DIM; r++) {
      memcpy(&tile_a[r * KTILE], &a[r * KDIM + k0], (size_t)KTILE);
    }
    bb_mvin((uintptr_t)tile_a, op1_bank_id, DIM * (KTILE / 16), 1);
    /* B rows k0..k0+KTILE-1 are contiguous: each row is DIM bytes. */
    bb_mvin((uintptr_t)(b + k0 * DIM), op2_bank_id, KTILE, 1);
    bb_transpose(op1_bank_id, a_transposed_bank_id, KTILE, 0);
    bb_mul_warp16(a_transposed_bank_id, op2_bank_id, acc_bank_id, KTILE, 0);
  }

  bb_mvout((uintptr_t)c, acc_bank_id, DIM, 1);
  bb_fence();
}

void init_diag_ones(elem_t *a, elem_t *b, result_t *expected) {
  clear_u8_matrix(a, DIM, KDIM);
  clear_u8_matrix(b, KDIM, DIM);
  clear_u32_matrix(expected, DIM, DIM);

  /* One 1 per k: A[r,k]=1 iff r==k%DIM, B[k,c]=1 iff c==k%DIM => C[r,c] nonzero
   * only on diagonal. */
  for (int k = 0; k < KDIM; k++) {
    int i = k % DIM;
    a[i * KDIM + k] = 1;
    b[k * DIM + i] = 1;
  }

  int diag_val = KDIM / DIM;
  for (int r = 0; r < DIM; r++) {
    expected[r * DIM + r] = diag_val;
  }
}

int run_test(const char *test_name, elem_t *a, elem_t *b) {
  hw_matmul_tiled(test_name, a, b, output_matrix);
  if (compare_u32_matrices(output_matrix, expected_matrix, DIM, DIM)) {
    printf("Test %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test %s FAILED\n", test_name);
    return 0;
  }
}

int test_tiled_matmul() {
  init_diag_ones(input_matrix_a, input_matrix_b, expected_matrix);
  return run_test("K-tiled diag-ones matmul", input_matrix_a, input_matrix_b);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_tiled_matmul();

  if (passed) {
    printf("tiled_matmul test PASSED\n");
    return 0;
  } else {
    printf("tiled_matmul test FAILED\n");
    return 1;
  }

#ifdef MULTICORE
  exit(0);
#endif
}
