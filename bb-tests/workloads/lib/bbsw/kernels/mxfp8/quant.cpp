#include "quant.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

static uint32_t roundedShift(uint32_t value, unsigned shift) {
  if (shift > 31)
    return 0;
  uint32_t half = uint32_t(1) << (shift - 1);
  return (value + half - 1 + ((value >> shift) & 1)) >> shift;
}

static uint8_t encode(uint32_t bits, int blockExponent) {
  uint8_t sign = (bits >> 24) & 128;
  if (!(bits & 0x7fffffff))
    return sign;
  int exponent = int((bits >> 23) & 255);
  uint32_t fraction = bits & 0x7fffff;
  if (exponent == 0) {
    int shift = __builtin_clz(fraction) - 8;
    fraction = (fraction << shift) & 0x7fffff;
    exponent = 1 - shift;
  }
  exponent -= blockExponent + 120;
  uint32_t code =
      exponent <= 0 ? roundedShift(fraction | 0x800000, unsigned(21 - exponent))
                    : unsigned(exponent * 8) + roundedShift(fraction, 20);
  return sign | std::min(code, 126u);
}

extern "C" void mxfp8_quant(const uint32_t *input, uint8_t *output,
                            uint32_t count) {
  for (uint32_t begin = 0; begin < count; begin += 32) {
    uint32_t maximum = 0;
    for (uint32_t i = 0; i < 32; ++i)
      maximum = std::max(maximum, input[begin + i] & 0x7fffffff);
    if (maximum >= 0x7f800000) {
      fputs("mxfp8: non-finite activation\n", stderr);
      abort();
    }
    int exponent = maximum == 0 ? 0 : std::max(-127, int(maximum >> 23) - 135);
    output[count + begin / 32] = uint8_t(exponent + 127);
    for (uint32_t i = 0; i < 32; ++i)
      output[begin + i] = encode(input[begin + i], exponent);
  }
}
