#include <bbhw/isa/isa.h>
#include <isa/mxint8quantize.h>
#include <stdint.h>
#include <stdio.h>

static float input[64] __attribute__((aligned(64)));
static int8_t output[96] __attribute__((aligned(64)));

int main(void) {
  for (unsigned i = 0; i < 64; ++i)
    input[i] = i % 2 ? -3.0f : 2.0f;
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 1, 16, 1);
  bb_mxint8quant(1, 2, 2, 32, 0, 0);
  bb_mvout((uintptr_t)output, 2, 6, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  for (unsigned i = 0; i < 96; ++i) {
    unsigned column = i % 48;
    int8_t expected = column < 32    ? (column % 2 ? -96 : 64)
                      : column == 32 ? 122
                                     : 0;
    if (output[i] != expected) {
      printf("Reduce quantization mismatch at %u: %d != %d\n", i, output[i],
             expected);
      return 1;
    }
  }
  return 0;
}
