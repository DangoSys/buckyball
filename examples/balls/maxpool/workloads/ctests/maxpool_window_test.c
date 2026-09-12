#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/maxpool.h>
#include <stdint.h>
#include <stdio.h>

enum { LANES = BANK_WIDTH / 8, INPUT_BASE = 1, OUTPUT_BASE = 2, STRIDE = 4 };
enum { INPUT_ROWS = INPUT_BASE + 36, OUTPUT_ROWS = OUTPUT_BASE + 11 };
static int8_t input[INPUT_ROWS * LANES] __attribute__((aligned(64)));
static int8_t output[OUTPUT_ROWS * LANES] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < INPUT_ROWS * LANES; ++i)
    input[i] = 23;
  for (int i = 0; i < OUTPUT_ROWS * LANES; ++i)
    output[i] = 23;
  for (int p = 0; p < 36; ++p)
    for (int c = 0; c < LANES; ++c)
      input[(INPUT_BASE + p) * LANES + c] = (int8_t)((p * 37 + c * 19) - 128);
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 0, INPUT_ROWS, 1);
  bb_mvin((uintptr_t)output, 1, OUTPUT_ROWS, 1);
  bb_maxpool(0, 1, 6, 3, 2, 2, 0, INPUT_BASE, OUTPUT_BASE, STRIDE, 0, 0);
  bb_mvout((uintptr_t)output, 1, OUTPUT_ROWS, 1);
  bb_fence();
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x)
      for (int c = 0; c < LANES; ++c) {
        int8_t expected = -128;
        for (int ky = 0; ky < 2; ++ky)
          for (int kx = 0; kx < 2; ++kx) {
            int8_t value =
                input[(INPUT_BASE + (y * 2 + ky) * 6 + x * 2 + kx) * LANES + c];
            if (value > expected)
              expected = value;
          }
        if (output[(OUTPUT_BASE + y * STRIDE + x) * LANES + c] != expected) {
          printf("maxpool window mismatch y=%d x=%d c=%d\n", y, x, c);
          return 1;
        }
      }
  printf("maxpool window PASS\n");
  return 0;
}
