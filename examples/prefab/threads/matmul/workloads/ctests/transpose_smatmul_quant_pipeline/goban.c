#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/quant.h>
#include <isa/smatmul.h>
#include <isa/transpose.h>
#include <multithread.h>
#include <stdint.h>

#if THREAD_MEM_BANK_WIDTH != 128 || THREAD_MEM_BANK_ENTRIES < 64
#error "Goban thread pipeline requires 128-bit banks with at least 64 rows"
#endif

enum { DIM = 16 };
static int8_t input[BB_NUM_CORES][DIM * DIM] __attribute__((aligned(64)));
static int8_t weight[BB_NUM_CORES][DIM * DIM] __attribute__((aligned(64)));
static int32_t bias[BB_NUM_CORES][DIM] __attribute__((aligned(64)));
static float scale[BB_NUM_CORES][DIM] __attribute__((aligned(64)));
static int8_t output[BB_NUM_CORES][DIM * DIM] __attribute__((aligned(64)));

#ifdef __cplusplus
extern "C"
#endif
    int goban_pipeline(uint32_t instance) {
  int slot = instance;
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col) {
      input[slot][row * DIM + col] = (row * 3 + col * 5) % 9 - 4;
      weight[slot][row * DIM + col] = (row + 2 * col) % 5 - 2;
    }
  for (int col = 0; col < DIM; ++col)
    scale[slot][col] = 1.0f;
  for (int bank = 0; bank < 10; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)input[slot], 0, DIM, 1);
  bb_mvin((uintptr_t)weight[slot], 2, DIM, 1);
  bb_mvin((uintptr_t)bias[slot], 3, 4, 1);
  bb_mvin((uintptr_t)scale[slot], 8, 4, 1);
  bb_transpose(0, 1, DIM, 8);
  bb_smatmul_bias(3, 0);
  bb_smatmul_os(1, 2, 4, DIM, DIM, DIM, 1, 1, 0);
  bb_quant_i32_to_i8(4, 8, 9, DIM * 4, 0, 0, 4, 4, 4, 0);
  bb_mvout((uintptr_t)output[slot], 9, DIM, 1);
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col) {
      int expected = 0;
      for (int k = 0; k < DIM; ++k)
        expected += input[slot][k * DIM + row] * weight[slot][k * DIM + col];
      if (output[slot][row * DIM + col] != expected)
        return 1;
    }
  return 0;
}
