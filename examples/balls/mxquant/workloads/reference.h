#ifndef MXQUANT_REFERENCE_H
#define MXQUANT_REFERENCE_H
#include <stdint.h>

/* Integer reference from bbsw/kernels/mxfp8/quant.cpp; no FP rounding state. */
static inline uint32_t rounded_shift(uint32_t value, unsigned shift) {
  if (shift > 31)
    return 0;
  uint32_t half = 1u << (shift - 1);
  return (value + half - 1 + ((value >> shift) & 1)) >> shift;
}
static inline void mxquant_reference32(const uint32_t *input,
                                       uint8_t *expected) {
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

#endif
