#ifndef SMATMUL_TEST_COMMON_H
#define SMATMUL_TEST_COMMON_H

#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_TILE 16
#define MATRIX_OUTPUT_GROUPS 2
#define MATRIX_OUTPUT_ROUNDS (4 / MATRIX_OUTPUT_GROUPS)

static inline void matrix_require_shape(int rows, int k) {
  if (rows <= 0 || k <= 0 || rows % MATRIX_TILE || k % MATRIX_TILE) {
    printf("smatmul requires non-zero 16-aligned rows and k\n");
    exit(1);
  }
}

static inline int matrix_a_lines(int rows, int k) {
  return rows * (k / MATRIX_TILE);
}

static inline void matrix_pack_a(const elem_t *src, elem_t *dst, int rows,
                                 int k) {
  int reduction_tiles = k / MATRIX_TILE;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < k; ++column) {
      int line = (row / MATRIX_TILE * reduction_tiles + column / MATRIX_TILE) *
                     MATRIX_TILE +
                 row % MATRIX_TILE;
      dst[line * MATRIX_TILE + column % MATRIX_TILE] = src[row * k + column];
    }
  }
}

static inline void matrix_pack_b(const elem_t *src, elem_t *dst, int k) {
  for (int row = 0; row < k; ++row)
    for (int column = 0; column < MATRIX_TILE; ++column)
      dst[row * MATRIX_TILE + column] = src[row * MATRIX_TILE + column];
}

static inline void matrix_pack_b_ws(const elem_t *src, elem_t *dst, int k,
                                    int n) {
  for (int panel = 0; panel < n / MATRIX_TILE; ++panel)
    for (int row = 0; row < k; ++row)
      for (int column = 0; column < MATRIX_TILE; ++column)
        dst[(panel * k + row) * MATRIX_TILE + column] =
            src[row * n + panel * MATRIX_TILE + column];
}

static inline void matrix_unpack_c(const result_t *src, result_t *dst,
                                   int rows) {
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < MATRIX_TILE; ++column) {
      int line = 2 * row + column / 8;
      dst[row * MATRIX_TILE + column] = src[line * 8 + column % 8];
    }
  }
}

static inline void matrix_unpack_c_ws(const result_t *src, result_t *dst,
                                      int rows, int columns) {
  for (int panel = 0; panel < columns / MATRIX_TILE; ++panel) {
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < MATRIX_TILE; ++column) {
        int line = panel * MATRIX_TILE * MATRIX_OUTPUT_ROUNDS +
                   row * MATRIX_OUTPUT_ROUNDS + column / 8;
        dst[row * columns + panel * MATRIX_TILE + column] =
            src[line * 8 + column % 8];
      }
    }
  }
}

static inline void matrix_hw_ws(const elem_t *a, const elem_t *b, result_t *c,
                                int rows, int columns, int k) {
  if (rows != MATRIX_TILE || columns <= 0 || columns % MATRIX_TILE ||
      columns > 512 || k != MATRIX_TILE)
    exit(1);
  const uint32_t a_bank = 0, b_bank = 1, c_bank = 2;
  bb_mem_alloc(a_bank, 1, 1);
  bb_mem_alloc(b_bank, 1, 1);
  bb_mem_alloc(c_bank, 1, MATRIX_OUTPUT_GROUPS);
  bb_mvin((uintptr_t)a, a_bank, rows, 1);
  bb_mvin((uintptr_t)b, b_bank, columns, 1);
  bb_smatmul_ws(a_bank, b_bank, c_bank, rows, columns, k);
  bb_mvout((uintptr_t)c, c_bank, columns * MATRIX_OUTPUT_ROUNDS, 1);
  bb_fence();
  bb_mem_release(a_bank);
  bb_mem_release(b_bank);
  bb_mem_release(c_bank);
}

static inline void matrix_hw(const elem_t *a, const elem_t *b, result_t *c,
                             int rows, int k) {
  const uint32_t a_bank = 0, b_bank = 1, c_bank = 2;
  matrix_require_shape(rows, k);
  bb_mem_alloc(a_bank, 1, 1);
  bb_mem_alloc(b_bank, 1, 1);
  bb_mem_alloc(c_bank, 1, 2);
  bb_mvin((uintptr_t)a, a_bank, matrix_a_lines(rows, k), 1);
  bb_mvin((uintptr_t)b, b_bank, k, 1);
  bb_smatmul_os(a_bank, b_bank, c_bank, rows, MATRIX_TILE, k);
  bb_mvout((uintptr_t)c, c_bank, 2 * rows, 1);
  bb_fence();
  bb_mem_release(a_bank);
  bb_mem_release(b_bank);
  bb_mem_release(c_bank);
}

#endif
