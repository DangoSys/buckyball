#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <isa/quant.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

static int8_t input[48] __attribute__((aligned(64)));
static int8_t weight[16 * 16] __attribute__((aligned(64)));
static int32_t bias[16] __attribute__((aligned(64)));
static float scale[16] __attribute__((aligned(64)));
static int8_t output[16 * 16] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < 36; ++i)
    input[i] = 1;
  for (int i = 36; i < 48; ++i)
    input[i] = 0;
  for (int row = 0; row < 16; ++row) {
    bias[row] = row - 8;
    scale[row] = 1.0f;
    for (int col = 0; col < 16; ++col)
      weight[row * 16 + col] = row < 9 ? 1 : 0;
  }

  for (int bank = 0; bank < 7; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)input, 0, 3, 1);
  bb_mvin((uintptr_t)weight, 2, 16, 1);
  bb_mvin((uintptr_t)bias, 3, 4, 1);
  bb_mvin((uintptr_t)scale, 5, 4, 1);
  bb_im2col(0, 1, 6, 3, 1, 0, 0, 0, 0, 0, 0, 16);
  bb_smatmul_bias(3);
  bb_smatmul_os(1, 2, 4, 16, 16, 16, 1, 1);
  bb_quant_i32_to_i8(4, 5, 6, 64, 0, 4, 4, 4, 0);
  bb_mvout((uintptr_t)output, 6, 16, 1);
  bb_fence();

  for (int window = 0; window < 16; ++window) {
    for (int channel = 0; channel < 16; ++channel) {
      int expected = 9 + bias[channel];
      int index = window * 16 + channel;
      if (output[index] != expected) {
        printf("mega_conv_pipeline FAIL window=%d channel=%d expected=%d "
               "actual=%d\n",
               window, channel, expected, output[index]);
        return 1;
      }
    }
  }
  printf("mega_conv_pipeline PASS\n");
  return 0;
}
