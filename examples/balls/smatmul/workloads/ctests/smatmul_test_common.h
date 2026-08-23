#ifndef MATRIX_TEST_COMMON_H
#define MATRIX_TEST_COMMON_H

#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_TILE 16
#define MATRIX_ACC_LANES 16

static inline int matrix_ceil_div(int x, int d) { return (x + d - 1) / d; }

static inline void matrix_require_cols(int n) {
  if (n < 1 || n > MATRIX_TILE) {
    printf("matrix: cols must be 1..%d, got %d\n", MATRIX_TILE, n);
    exit(1);
  }
}

static inline int matrix_a_rows(int m, int k) {
  return matrix_ceil_div(m, MATRIX_TILE) * matrix_ceil_div(k, MATRIX_TILE) *
         MATRIX_TILE;
}

static inline int matrix_b_rows(int n, int k) {
  matrix_require_cols(n);
  return matrix_ceil_div(k, MATRIX_TILE) * MATRIX_TILE;
}

static inline int matrix_c_blocks(int m, int n) {
  matrix_require_cols(n);
  return m;
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
      int lane = c % MATRIX_TILE;
      int kti = r / MATRIX_TILE;
      int kr = r % MATRIX_TILE;
      int bank_row = kti * MATRIX_TILE + kr;
      dst[bank_row * MATRIX_TILE + lane] = src[r * n + c];
    }
  }
}

static inline void matrix_unpack_c(const result_t *src, result_t *dst, int m,
                                   int n) {
  matrix_require_cols(n);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c)
      dst[r * n + c] = src[r * MATRIX_ACC_LANES + c];
  }
}

static inline void matrix_issue(const elem_t *packed_a, const elem_t *packed_b,
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
  if (ws)
    bb_smatmul_ws(op1, op2, wr, m, n, k);
  else
    bb_smatmul_os(op1, op2, wr, m, n, k);
  bb_mvout((uintptr_t)packed_c, wr, c_blocks, 1);
  bb_fence();
  bb_mem_release(op1);
  bb_mem_release(op2);
  bb_mem_release(wr);
}

static inline void matrix_hw_os(const elem_t *packed_a, const elem_t *packed_b,
                                result_t *packed_c, int m, int n, int k) {
  matrix_issue(packed_a, packed_b, packed_c, m, n, k, 0);
}

static inline void matrix_hw_ws(const elem_t *packed_a, const elem_t *packed_b,
                                result_t *packed_c, int m, int n, int k) {
  matrix_issue(packed_a, packed_b, packed_c, m, n, k, 1);
}

#endif
