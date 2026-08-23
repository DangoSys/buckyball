#include "pebble.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdio.h>

#ifndef TEST_CASE_NAME
#define TEST_CASE_NAME "conv_6x6_k3"
#endif

enum {
  MAX_WIN = 11 * 11,
  MAX_K = 7 * 7,
  MAX_IM2COL = ((MAX_WIN + 15) / 16) * ((MAX_K + 15) / 16) * 16 * 16,
};

static elem_t matrix_a[MAX_IM2COL] __attribute__((aligned(64)));
static elem_t matrix_b[MAX_K * PEBBLE_INT8_LANES] __attribute__((aligned(64)));
static result_t packed_c[MAX_WIN * PEBBLE_ACC_LANES]
    __attribute__((aligned(64)));

static int phys(int row, int col, int k_tiles) {
  int m_tile = row / PEBBLE_INT8_LANES;
  int m_row = row % PEBBLE_INT8_LANES;
  int k_tile = col / PEBBLE_INT8_LANES;
  int lane = col % PEBBLE_INT8_LANES;
  int bank_row = (m_tile * k_tiles + k_tile) * PEBBLE_INT8_LANES + m_row;
  return bank_row * PEBBLE_INT8_LANES + lane;
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  const pebble_conv_test_case_t *t = pebble_find_conv_test_case(TEST_CASE_NAME);
  if (!t) {
    printf("unknown case %s\n", TEST_CASE_NAME);
    return 1;
  }

  int wins = t->output_h * t->output_w;
  int k_elems = t->kernel_h * t->kernel_w;
  int k_tiles = (k_elems + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;
  int m_tiles = (wins + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;
  int a_rows = m_tiles * k_tiles * PEBBLE_INT8_LANES;

  for (int i = 0; i < MAX_IM2COL; ++i)
    matrix_a[i] = 0;
  for (int row = 0; row < wins; ++row)
    for (int col = 0; col < k_elems; ++col)
      matrix_a[phys(row, col, k_tiles)] =
          t->expected_im2col[row * k_elems + col];

  for (int i = 0; i < MAX_K * PEBBLE_INT8_LANES; ++i)
    matrix_b[i] = 0;
  for (int row = 0; row < k_elems; ++row)
    matrix_b[row * PEBBLE_INT8_LANES] = t->kernel[row];
  bb_dma_touch(packed_c, sizeof(packed_c));

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  bb_mvin((uintptr_t)matrix_a, 0, a_rows, 1);
  bb_mvin((uintptr_t)matrix_b, 1, k_elems, 1);
  bb_smatmul_compiled(0, 1, 2, wins, 1, k_elems);
  bb_mvout((uintptr_t)packed_c, 2, wins, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);

  int passed = 1;
  for (int i = 0; i < wins; ++i) {
    result_t got = packed_c[i * PEBBLE_ACC_LANES];
    if (got != t->expected_output[i]) {
      printf("FAIL matmul i=%d exp=%d got=%d\n", i, (int)t->expected_output[i],
             (int)got);
      passed = 0;
    }
  }
  printf("conv_matmul [%s] %s\n", t->name, passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
