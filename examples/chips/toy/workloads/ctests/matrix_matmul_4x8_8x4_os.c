#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define M_DIM 4
#define K_DIM 8
#define N_DIM 4
#define TILE_DIM 4
#define PACKED_IN_COLS (BANK_WIDTH / 8 / (int)sizeof(elem_t))
#define PACKED_OUT_COLS (4 * (BANK_WIDTH / 8 / (int)sizeof(result_t)))

static elem_t a[M_DIM * K_DIM] __attribute__((aligned(64)));
static elem_t b[K_DIM * N_DIM] __attribute__((aligned(64)));
static result_t expected[M_DIM * N_DIM] __attribute__((aligned(64)));
static result_t output[M_DIM * N_DIM] __attribute__((aligned(64)));
static elem_t packed_a[2][TILE_DIM * PACKED_IN_COLS]
    __attribute__((aligned(64)));
static elem_t packed_b[2][TILE_DIM * PACKED_IN_COLS]
    __attribute__((aligned(64)));
static result_t packed_c[TILE_DIM * PACKED_OUT_COLS]
    __attribute__((aligned(64)));

static void pack_a(const elem_t *src, int k0, elem_t *dst) {
  clear_u8_matrix(dst, TILE_DIM, PACKED_IN_COLS);
  for (int r = 0; r < TILE_DIM; ++r)
    for (int c = 0; c < TILE_DIM; ++c)
      dst[r * PACKED_IN_COLS + c] = src[r * K_DIM + k0 + c];
}

static void pack_b(const elem_t *src, int k0, elem_t *dst) {
  clear_u8_matrix(dst, TILE_DIM, PACKED_IN_COLS);
  for (int r = 0; r < TILE_DIM; ++r)
    for (int c = 0; c < TILE_DIM; ++c)
      dst[r * PACKED_IN_COLS + c] = src[(k0 + r) * N_DIM + c];
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  init_u8_random_matrix(a, M_DIM, K_DIM, 123);
  init_u8_random_matrix(b, K_DIM, N_DIM, 456);
  clear_u32_matrix(output, M_DIM, N_DIM);
  clear_u32_matrix(expected, M_DIM, N_DIM);
  clear_u32_matrix(packed_c, TILE_DIM, PACKED_OUT_COLS);
  cpu_matmul(a, b, expected, M_DIM, N_DIM, K_DIM);

  pack_a(a, 0, packed_a[0]);
  pack_a(a, 4, packed_a[1]);
  pack_b(b, 0, packed_b[0]);
  pack_b(b, 4, packed_b[1]);

  const uint32_t op1 = 0, op2 = 1, acc = 2;
  bb_mem_alloc(op1, 1, 1);
  bb_mem_alloc(op2, 1, 1);
  bb_mem_alloc(acc, 1, 4);
  bb_mvin((uintptr_t)packed_a[0], op1, TILE_DIM, 1);
  bb_mvin((uintptr_t)packed_b[0], op2, TILE_DIM, 1);
  bb_matrix_os_ACC_FIRST(op1, op2, acc, TILE_DIM);
  bb_mvin((uintptr_t)packed_a[1], op1, TILE_DIM, 1);
  bb_mvin((uintptr_t)packed_b[1], op2, TILE_DIM, 1);
  bb_matrix_os_ACC_LAST(op1, op2, acc, TILE_DIM);
  bb_mvout((uintptr_t)packed_c, acc, TILE_DIM, 1);
  bb_fence();

  for (int r = 0; r < TILE_DIM; ++r)
    for (int c = 0; c < TILE_DIM; ++c)
      output[r * N_DIM + c] = packed_c[r * PACKED_OUT_COLS + c * 4];

  int passed = compare_u32_matrices(output, expected, M_DIM, N_DIM);
  printf("matrix_matmul_4x8_8x4_os %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
