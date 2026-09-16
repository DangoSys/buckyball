#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum {
  ROWS = 16,
  SOURCE_WIDTH = 56,
  PIXEL_BYTES = 16,
  SOURCE_ROW_BYTES = SOURCE_WIDTH * PIXEL_BYTES,
  INPUT_OFFSET = 3 * SOURCE_ROW_BYTES + 160,
  ITERATIONS = 30000
};

static uint8_t arena[INPUT_OFFSET + ROWS * PIXEL_BYTES]
    __attribute__((aligned(64)));
static int8_t lhs[16] __attribute__((aligned(64)));
static int32_t bias[16] __attribute__((aligned(64)));
static uint8_t output[ROWS * PIXEL_BYTES] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < (int)sizeof(arena); ++i)
    arena[i] = (uint8_t)(i * 17 + 3);
  for (int i = 0; i < 16; ++i)
    lhs[i] = (int8_t)(i % 5 - 2);

  for (int bank = 0; bank <= 8; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)lhs, 4, 1, 1);
  bb_mvin((uintptr_t)bias, 5, 4, 1);
  bb_mvin((uintptr_t)&arena[INPUT_OFFSET], 8, ROWS, 1);
  bb_smatmul_bias(5, 0);
  for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
    bb_mem_release(7);
    bb_smatmul_os(4, 8, 6, 1, 16, 16, 1, 1, 0);
    bb_mem_release(8);
    bb_mem_alloc(7, 1, 1);
    bb_mvin((uintptr_t)&arena[INPUT_OFFSET], 7, ROWS, 1);
    bb_mem_alloc(8, 1, 1);
    bb_mvin((uintptr_t)&arena[INPUT_OFFSET], 8, ROWS, 1);
    for (int chunk = 0; chunk < 4; ++chunk)
      bb_mvin_2d((uintptr_t)&arena[chunk * SOURCE_ROW_BYTES], 8, 1, PIXEL_BYTES,
                 SOURCE_WIDTH, chunk * 4, 4, 16);
  }
  bb_mvout((uintptr_t)output, 8, ROWS, 1);
  bb_fence();

  for (int chunk = 0; chunk < 4; ++chunk)
    for (int pixel = 0; pixel < 4; ++pixel)
      for (int byte = 0; byte < PIXEL_BYTES; ++byte) {
        int output_index = (chunk * 4 + pixel) * PIXEL_BYTES + byte;
        int source_index =
            chunk * SOURCE_ROW_BYTES + pixel * PIXEL_BYTES + byte;
        if (output[output_index] != arena[source_index]) {
          printf("mvin2d stress mismatch chunk=%d pixel=%d byte=%d got=%u "
                 "expected=%u\n",
                 chunk, pixel, byte, output[output_index], arena[source_index]);
          return 1;
        }
      }
  printf("mvin bank7/8 stress PASS iterations=%d\n", ITERATIONS);
  return 0;
}
