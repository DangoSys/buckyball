#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <isa/quant.h>
#include <isa/smatmul.h>
#include <multicore.h>
#include <stdint.h>

#if BANK_WIDTH != 128 || BANK_LINES < 64
#error "Pebble core pipeline requires 128-bit banks with at least 64 rows"
#endif

enum { LANES = BANK_WIDTH / 8, INPUT_ROWS = 36, WINDOWS = 16 };
static int8_t input[BB_CORES_PER_TILE][INPUT_ROWS * LANES]
    __attribute__((aligned(64)));
static int8_t weight[BB_CORES_PER_TILE][LANES * LANES]
    __attribute__((aligned(64)));
static int32_t bias[BB_CORES_PER_TILE][LANES] __attribute__((aligned(64)));
static float scale[BB_CORES_PER_TILE][LANES] __attribute__((aligned(64)));
static int8_t output[BB_CORES_PER_TILE][WINDOWS * LANES]
    __attribute__((aligned(64)));

#ifdef __cplusplus
extern "C"
#endif
    int pebble_core(core_id_t id) {
  int core = id.core;
  for (int row = 0; row < INPUT_ROWS; ++row)
    input[core][row * LANES] = (3 * row + 1) % 7 - 3;
  for (int channel = 0; channel < LANES; ++channel) {
    bias[core][channel] = channel - 8;
    scale[core][channel] = 1.0f;
    for (int k = 0; k < LANES; ++k)
      weight[core][k * LANES + channel] =
          k < 9 ? (2 * k + 3 * channel) % 5 - 2 : 0;
  }
  for (int bank = 0; bank < 7; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)input[core], 0, INPUT_ROWS, 1);
  bb_mvin((uintptr_t)weight[core], 2, LANES, 1);
  bb_mvin((uintptr_t)bias[core], 3, LANES * 32 / BANK_WIDTH, 1);
  bb_mvin((uintptr_t)scale[core], 5, LANES * 32 / BANK_WIDTH, 1);
  bb_im2col(0, 1, 6, 3, 1, 0, 0, 0, 0, 0, 0, WINDOWS);
  bb_smatmul_bias(3, 0);
  bb_smatmul_os(1, 2, 4, WINDOWS, LANES, LANES, 1, 1, 0);
  bb_quant_i32_to_i8(4, 5, 6, WINDOWS * 4, 0, 0, 4, 4, 4, 0);
  bb_mvout((uintptr_t)output[core], 6, WINDOWS, 1);
  bb_fence();
  for (int window = 0; window < WINDOWS; ++window)
    for (int channel = 0; channel < LANES; ++channel) {
      int expected = bias[core][channel];
      int y = window / 4, x = window % 4;
      for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx) {
          int k = ky * 3 + kx, row = (y + ky) * 6 + x + kx;
          expected +=
              input[core][row * LANES] * weight[core][k * LANES + channel];
        }
      if (output[core][window * LANES + channel] != expected)
        return 1;
    }
  return 0;
}
