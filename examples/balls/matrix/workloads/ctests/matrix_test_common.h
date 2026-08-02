#ifndef MATRIX_TEST_COMMON_H
#define MATRIX_TEST_COMMON_H

#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#define MATRIX_TILE 16
#define MATRIX_ACC_LANES 16

static inline int matrix_ceil_div(int x, int d) { return (x + d - 1) / d; }

static inline int matrix_a_rows(int m, int k) {
  return matrix_ceil_div(m, MATRIX_TILE) * matrix_ceil_div(k, MATRIX_TILE) *
         MATRIX_TILE;
}

static inline int matrix_b_rows(int n, int k) {
  return matrix_ceil_div(n, MATRIX_TILE) * matrix_ceil_div(k, MATRIX_TILE) *
         MATRIX_TILE;
}

static inline int matrix_c_blocks(int m, int n) {
  return m * matrix_ceil_div(n, MATRIX_TILE);
}

static inline void matrix_pack_a(const elem_t *src, elem_t *dst, int m, int k) {
  int kt = matrix_ceil_div(k, MATRIX_TILE);
  int rows = matrix_a_rows(m, k);
  for (int i = 0; i < rows * MATRIX_TILE; ++i)
    dst[i] = 0;
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < k; ++c) {
      int mt = r / MATRIX_TILE;
      int mr = r % MATRIX_TILE;
      int kti = c / MATRIX_TILE;
      int lane = c % MATRIX_TILE;
      int bank_row = (mt * kt + kti) * MATRIX_TILE + mr;
      dst[bank_row * MATRIX_TILE + lane] = src[r * k + c];
    }
  }
}

static inline void matrix_pack_b(const elem_t *src, elem_t *dst, int k, int n) {
  int kt = matrix_ceil_div(k, MATRIX_TILE);
  int rows = matrix_b_rows(n, k);
  for (int i = 0; i < rows * MATRIX_TILE; ++i)
    dst[i] = 0;
  for (int r = 0; r < k; ++r) {
    for (int c = 0; c < n; ++c) {
      int nt = c / MATRIX_TILE;
      int lane = c % MATRIX_TILE;
      int kti = r / MATRIX_TILE;
      int kr = r % MATRIX_TILE;
      int bank_row = (nt * kt + kti) * MATRIX_TILE + kr;
      dst[bank_row * MATRIX_TILE + lane] = src[r * n + c];
    }
  }
}

static inline int matrix_c_block(int row, int n_tile, int m, int n) {
  int n_tiles = matrix_ceil_div(n, MATRIX_TILE);
  int mt = row / MATRIX_TILE;
  int mr = row % MATRIX_TILE;
  int block = 0;
  for (int t = 0; t < mt; ++t) {
    int rows = m - t * MATRIX_TILE;
    if (rows > MATRIX_TILE)
      rows = MATRIX_TILE;
    block += rows * n_tiles;
  }
  int rows = m - mt * MATRIX_TILE;
  if (rows > MATRIX_TILE)
    rows = MATRIX_TILE;
  return block + n_tile * rows + mr;
}

static inline void matrix_unpack_c(const result_t *src, result_t *dst, int m,
                                   int n) {
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      int nti = c / MATRIX_TILE;
      int lane = c % MATRIX_TILE;
      int block = matrix_c_block(r, nti, m, n);
      dst[r * n + c] = src[block * MATRIX_ACC_LANES + lane];
    }
  }
}

static inline void matrix_hw_mnk(const elem_t *packed_a, const elem_t *packed_b,
                                 result_t *packed_c, int m, int n, int k,
                                 int ws) {
  uint32_t op1 = 0, op2 = 1, wr = 2;
  int a_rows = matrix_a_rows(m, k);
  int b_rows = matrix_b_rows(n, k);
  int c_blocks = matrix_c_blocks(m, n);

  bb_mem_alloc(op1, 1, 1);
  bb_mem_alloc(op2, 1, 1);
  bb_mem_alloc(wr, 1, 4);
  bb_mvin((uintptr_t)packed_a, op1, a_rows, 1);
  bb_mvin((uintptr_t)packed_b, op2, b_rows, 1);
  bb_matrix_mnk_mode(op1, op2, wr, m, n, k,
                     ws ? BB_MATRIX_MODE_WS : BB_MATRIX_MODE_OS);
  bb_mvout((uintptr_t)packed_c, wr, c_blocks, 1);
  bb_fence();
}

#endif
