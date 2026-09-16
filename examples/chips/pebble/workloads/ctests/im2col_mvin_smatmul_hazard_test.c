#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/im2col.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

static int8_t input[4096 * 16] __attribute__((aligned(128)));
static int8_t pixel_input[28 * 24] __attribute__((aligned(128)));
static int8_t weights[16 * 16] __attribute__((aligned(128)));
static const int32_t bias[16] __attribute__((aligned(128)));
static int8_t output[16 * 16] __attribute__((aligned(128)));

int main(void) {
  const int iterations = 1;
  for (size_t i = 0; i < sizeof(input); ++i)
    input[i] = (int8_t)((i * 13 + 7) % 17 - 8);
  for (size_t i = 0; i < sizeof(pixel_input); ++i)
    pixel_input[i] = (int8_t)((i * 5 + 3) % 11 - 5);
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col)
      weights[row * 16 + col] = row == col ? 1 : 0;
  for (int bank = 1; bank <= 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_release(5);
  for (int bank = 5; bank <= 8; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_release(8);
  bb_mem_release(5);
  bb_mem_release(6);
  bb_mem_alloc(5, 1, 1);
  bb_mem_alloc(6, 1, 1);
  bb_mem_release(7);
  bb_mem_release(5);
  bb_mem_alloc(5, 1, 1);
  bb_mem_release(6);
  bb_mem_release(5);
  bb_mem_release(4);
  bb_mem_alloc(4, 1, 1);
  bb_mem_alloc(5, 1, 1);
  bb_mem_release(5);
  for (int bank = 5; bank <= 8; ++bank)
    bb_mem_alloc(bank, 1, 1);

  bb_mvin((uintptr_t)bias, 4, 4, 1);
  bb_smatmul_bias(4, 0);
  bb_mvin((uintptr_t)input, 8, 18, 1);
  bb_mvin((uintptr_t)input, 8, 4096, 1);
  bb_mvin_2d((uintptr_t)pixel_input, 8, 1, 24, 28, 6, 3, 16);
  bb_mvin_2d((uintptr_t)(pixel_input + 16), 8, 1, 24, 28, 15, 3, 8);
  for (int iteration = 0; iteration < iterations; ++iteration)
    for (int lane = 0; lane < 16; ++lane) {
      bb_im2col(8, 5, 3, 1, 1, 0, 0, lane, 0, 0, 0, 9);
      bb_mvin((uintptr_t)weights, 6, 16, 1);
      bb_smatmul_os(5, 6, 7, 16, 16, 16, lane == 0, lane == 15, 0);
    }
  bb_mvout((uintptr_t)output, 7, 16, 1);
  bb_fence();

  uint32_t hash = 2166136261U;
  for (size_t i = 0; i < sizeof(output); ++i) {
    hash ^= (uint8_t)output[i];
    hash *= 16777619U;
  }
  printf("im2col mvin smatmul hazard output hash=%08x sample=", hash);
  for (int i = 0; i < 16; ++i)
    printf("%d%s", output[i], i == 15 ? "\n" : ",");
  printf("im2col mvin smatmul hazard PASS iterations=%d\n", iterations);
  return 0;
}
