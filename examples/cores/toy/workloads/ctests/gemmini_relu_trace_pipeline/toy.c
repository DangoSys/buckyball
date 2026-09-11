#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/bdb_counter.h>
#include <isa/gemmini.h>
#include <isa/relu.h>
#include <multicore.h>
#include <stdint.h>

#if BANK_WIDTH != 128 || BANK_LINES < 64 || BANK_LINES % 64 != 0
#error "Toy core pipeline requires 128-bit banks and depth divisible by 64"
#endif

enum { DIM = 16 };
static int8_t input[DIM * DIM] __attribute__((aligned(64)));
static int8_t weight[DIM * DIM] __attribute__((aligned(64)));
static int8_t zeros[DIM * DIM] __attribute__((aligned(64)));
static int32_t output[DIM * DIM] __attribute__((aligned(64)));

#ifdef __cplusplus
extern "C"
#endif
int toy_core(core_id_t id) {
  (void)id;
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col) {
      input[row * DIM + col] = (row * 3 + col * 5) % 15 - 7;
      weight[row * DIM + col] = (row + 2 * col) % 5 - 2;
    }
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)input, 0, DIM, 1);
  bb_mvin((uintptr_t)weight, 1, DIM, 1);
  bb_mvin((uintptr_t)zeros, 3, DIM, 1);
  bdb_counter_start(0, 0xC001);
  bb_gemmini_config(0, 0, 0, 0, 0);
  bb_gemmini_preload(3, 2, DIM, 0, 0);
  bb_gemmini_compute_preloaded(0, 1, 2, DIM, 0, 0, 0);
  for (int group = 0; group < 4; ++group)
    bb_relu(2, group, DIM * 4, DIM * 4);
  bdb_counter_stop(0);
  bb_mvout((uintptr_t)output, 2, DIM, 1);
  bb_fence();
  for (int i = 0; i < DIM * DIM; ++i) {
    int row = i / DIM, col = i % DIM;
    int32_t expected = 0;
    for (int k = 0; k < DIM; ++k)
      expected += input[row * DIM + k] * weight[k * DIM + col];
    expected = expected < 0 ? 0 : expected;
    if (output[i] != expected)
      return 1;
  }
  return 0;
}
