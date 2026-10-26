#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <isa/im2col.h>
#include <isa/maxpool.h>
#include <isa/quant.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum { INPUT_SIDE = 8, OUTPUT_SIDE = 4, KERNEL = 5, CORE = 16, PADDED_K = 32 };
static int8_t input[INPUT_SIDE * INPUT_SIDE * CORE]
    __attribute__((aligned(64)));
static int8_t weight[PADDED_K * CORE] __attribute__((aligned(64)));
static int32_t bias[CORE] __attribute__((aligned(64)));
static float scale[CORE] __attribute__((aligned(64)));
static int8_t output[OUTPUT_SIDE * OUTPUT_SIDE * CORE]
    __attribute__((aligned(64)));

int main(void) {
  for (int pixel = 0; pixel < INPUT_SIDE * INPUT_SIDE; ++pixel)
    input[pixel * CORE] = pixel % 5 - 2;
  for (int k = 0; k < KERNEL * KERNEL; ++k)
    for (int channel = 0; channel < CORE; ++channel)
      weight[k * CORE + channel] = (k + channel) % 3 - 1;
  for (int channel = 0; channel < CORE; ++channel)
    scale[channel] = 1.0f;

  for (int bank = 0; bank < 8; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)input, 0, INPUT_SIDE * INPUT_SIDE, 1);
  bb_mvin((uintptr_t)weight, 2, PADDED_K, 1);
  bb_mvin((uintptr_t)bias, 3, 4, 1);
  bb_mvin((uintptr_t)scale, 5, 4, 1);
  bb_im2col(0, 1, INPUT_SIDE, KERNEL, 1, 0, 0, 0, 0, 0, 0,
            OUTPUT_SIDE * OUTPUT_SIDE);
  for (int row_core = 0; row_core < 2; ++row_core)
    for (int k_core = 0; k_core < 2; ++k_core)
      for (int row = 0; row < 8; ++row) {
        int input_row = k_core * 16 + row_core * 8 + row;
        int output_row = (row_core * 2 + k_core) * 8 + row;
        bb_maxpool(1, 7, 1, 1, 1, 1, 0, input_row, output_row, 1, 0, 0);
      }
  bb_smatmul_bias(3, 0);
  bb_smatmul_os(7, 2, 4, OUTPUT_SIDE * OUTPUT_SIDE, CORE, PADDED_K, 1, 1, 0);
  bb_quant_i32_to_i8(4, 5, 6, OUTPUT_SIDE * OUTPUT_SIDE * 4, 0, 0, OUTPUT_SIDE,
                     OUTPUT_SIDE, OUTPUT_SIDE, 0);
  bb_mvout((uintptr_t)output, 6, OUTPUT_SIDE * OUTPUT_SIDE, 1);

  for (int y = 0; y < OUTPUT_SIDE; ++y)
    for (int x = 0; x < OUTPUT_SIDE; ++x)
      for (int channel = 0; channel < CORE; ++channel) {
        int expected = 0;
        for (int ky = 0; ky < KERNEL; ++ky)
          for (int kx = 0; kx < KERNEL; ++kx) {
            int pixel = (y + ky) * INPUT_SIDE + x + kx;
            int k = ky * KERNEL + kx;
            expected += input[pixel * CORE] * weight[k * CORE + channel];
          }
        int index = (y * OUTPUT_SIDE + x) * CORE + channel;
        if (output[index] != expected) {
          printf(
              "mega_conv_k5 FAIL y=%d x=%d channel=%d expected=%d actual=%d\n",
              y, x, channel, expected, output[index]);
          return 1;
        }
      }
  printf("mega_conv_k5 PASS\n");
  return 0;
}
