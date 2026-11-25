// mxquant on the thread alone: 32 FP32 values to 32 MXFP8 codes and their
// block exponent, for signed zeros, edges, ties, subnormals and 4 random
// blocks, checked against the integer reference of the Ball (copied from its
// workloads/reference.h). Only the 32 codes and the exponent byte are
// checked: the Ball's write defines the rest of the exponent row.
#include <bbhw/isa/isa.h>
#include <isa/mxquant.h>
#include <stdint.h>

#define INPUT ((volatile uint32_t *)HOST_TSS_BASE)
#define OUTPUT ((volatile uint8_t *)(HOST_TSS_BASE + 128))

static uint32_t rounded_shift(uint32_t value, unsigned shift) {
  if (shift > 31)
    return 0;
  uint32_t half = 1u << (shift - 1);
  return (value + half - 1 + ((value >> shift) & 1)) >> shift;
}

static void reference(const uint32_t *input, uint8_t *expected) {
  uint32_t maximum = 0;
  for (int i = 0; i < 32; ++i) {
    uint32_t magnitude = input[i] & 0x7fffffffu;
    if (magnitude > maximum)
      maximum = magnitude;
  }
  int block = maximum == 0 ? 0 : (int)(maximum >> 23) - 135;
  if (block < -127)
    block = -127;
  expected[32] = (uint8_t)(block + 127);
  for (int i = 0; i < 32; ++i) {
    uint32_t bits = input[i], fraction = bits & 0x7fffffu;
    uint8_t sign = (bits >> 24) & 128;
    if (!(bits & 0x7fffffffu)) {
      expected[i] = sign;
      continue;
    }
    int exponent = (bits >> 23) & 255;
    if (!exponent) {
      for (exponent = 1; !(fraction & 0x800000u); --exponent)
        fraction <<= 1;
      fraction &= 0x7fffffu;
    }
    exponent -= block + 120;
    uint32_t code = exponent <= 0
                        ? rounded_shift(fraction | 0x800000u, 21 - exponent)
                        : exponent * 8 + rounded_shift(fraction, 20);
    expected[i] = sign | (code < 126 ? code : 126);
  }
}

static const uint32_t edges[] = {0,          0x80000000, 1,          0x80000001,
                                 0x007fffff, 0x00800000, 0x3f880000, 0x3f980000,
                                 0x43e00000, 0x7f7fffff, 0xff7fffff};
static const uint32_t subnormals[] = {0,      1,        0x1000,     0x2000,
                                      0x3000, 0x7fffff, 0x80000001, 0x807fffff};

uint64_t entry(void) {
  uint32_t input[32];
  uint8_t expected[64];
  uint32_t random = 0x6d2b79f5;
  for (int test = 0; test < 8; ++test) {
    for (int i = 0; i < 32; ++i) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      input[i] =
          test == 0   ? (i & 1 ? 0x80000000u : 0)
          : test == 1 ? edges[i % 11]
          : test == 2 ? (i ? (i & 1 ? 0x3f880000u : 0x3f980000u) : 0x43e00000u)
          : test == 3 ? subnormals[i % 8]
                      : ((random & 0x807fffffu) |
                         ((((test - 4) * 4 + (random >> 23 & 3)) % 255) << 23));
      INPUT[i] = input[i];
    }
    for (int i = 0; i < 64; ++i) {
      expected[i] = 0x5a;
      OUTPUT[i] = 0x5a;
    }
    reference(input, expected);
    bb_mem_alloc(3, 1, 1);
    bb_mem_alloc(4, 1, 1);
    bb_mvin((uintptr_t)INPUT, 3, 8, 1);
    bb_mvin((uintptr_t)OUTPUT, 4, 4, 1);
    bb_mxquant(3, 4, 32);
    bb_mvout((uintptr_t)OUTPUT, 4, 4, 1);
    asm volatile("fence" ::: "memory");
    for (int i = 0; i < 33; ++i)
      if (OUTPUT[i] != expected[i])
        return 1 + test * 64 + i;
    bb_mem_release(3);
    bb_mem_release(4);
  }
  return 0;
}
