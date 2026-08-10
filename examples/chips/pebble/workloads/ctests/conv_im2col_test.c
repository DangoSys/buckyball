#include "pebble.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

#ifndef TEST_CASE_NAME
#define TEST_CASE_NAME "conv_6x6_k3"
#endif

enum {
  MAX_IN = 17 * 17,
  MAX_WIN = 11 * 11,
  MAX_K = 7 * 7,
  MAX_IM2COL = ((MAX_WIN + 15) / 16) * ((MAX_K + 15) / 16) * 16 * 16,
};

static elem_t packed_in[(MAX_IN + 15) / 16 * 16] __attribute__((aligned(64)));
static elem_t actual[MAX_IM2COL] __attribute__((aligned(64)));

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

  int in_elems = t->input_h * t->input_w;
  int wins = t->output_h * t->output_w;
  int k_elems = t->kernel_h * t->kernel_w;
  int k_tiles = (k_elems + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;
  int m_tiles = (wins + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;
  int beats = (in_elems + PEBBLE_INT8_LANES - 1) / PEBBLE_INT8_LANES;
  int rows = m_tiles * k_tiles * PEBBLE_INT8_LANES;

  for (int i = 0; i < (int)(sizeof(packed_in) / sizeof(packed_in[0])); ++i)
    packed_in[i] = 0;
  for (int i = 0; i < in_elems; ++i)
    packed_in[i] = t->input[i];
  bb_dma_touch(actual, sizeof(actual));

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)packed_in, 0, beats, 1);
  bb_im2col(0, 1, t->input_h, t->kernel_h, t->stride, t->padding);
  bb_mvout((uintptr_t)actual, 1, rows, 1);
  bb_fence();

  int passed = 1;
  for (int row = 0; row < m_tiles * PEBBLE_INT8_LANES; ++row) {
    for (int col = 0; col < k_tiles * PEBBLE_INT8_LANES; ++col) {
      elem_t got = actual[phys(row, col, k_tiles)];
      elem_t exp = (row < wins && col < k_elems)
                       ? t->expected_im2col[row * k_elems + col]
                       : 0;
      if (got != exp) {
        printf("FAIL im2col row=%d col=%d exp=%d got=%d\n", row, col, (int)exp,
               (int)got);
        passed = 0;
      }
    }
  }
  printf("conv_im2col [%s] %s\n", t->name, passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}
