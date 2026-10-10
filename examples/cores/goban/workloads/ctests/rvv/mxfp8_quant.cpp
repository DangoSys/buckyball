#include <bbhw/isa/isa.h>
#include <bbsw/kernels/mxfp8/quant.h>
#include <cmath>
#include <cstdio>
#include <cstring>

alignas(16) static float input[128];
alignas(16) static uint8_t output[144];
static double candidates[127];

// Enumerate finite E4M3 codes independently, choosing nearest, ties to even.
static uint8_t encode(float value, int scale) {
  double magnitude = std::ldexp(std::fabs(double(value)), -scale);
  double distance = magnitude;
  unsigned selected = 0;
  for (unsigned code = 1; code <= 126; ++code) {
    double error = std::fabs(magnitude - candidates[code]);
    if (error < distance || (error == distance && !(code & 1))) {
      selected = code;
      distance = error;
    }
  }
  return selected | (std::signbit(value) ? 128 : 0);
}

int main() {
  for (unsigned code = 1; code <= 126; ++code) {
    unsigned exponent = code >> 3, fraction = code & 7;
    candidates[code] = exponent
                           ? std::ldexp(1.0 + fraction / 8.0, int(exponent) - 7)
                           : std::ldexp(double(fraction), -9);
  }
  for (unsigned i = 0; i < 32; ++i) {
    input[i] = (int(i) - 16) * 0.25f;
    input[32 + i] = (i & 1 ? -1.0f : 1.0f) * (16.0f + i * 0.5f);
    input[64 + i] = i & 1 ? -0.0f : 0.0f;
    input[96 + i] = std::ldexp(float(int(i) - 16), -145);
  }
  // Scale 0; these small lanes exercise FP8 subnormals and ties to even.
  input[0] = 256.0f;
  input[1] = 0x1p-10f;
  input[2] = 0x1.8p-9f;
  input[3] = -0x1p-9f;
  input[31] = -448.0f;
  input[63] = 31.0f;
  mxfp8_quant(reinterpret_cast<const uint32_t *>(input), output, 128);
  for (unsigned block = 0; block < 4; ++block) {
    float maximum = 0;
    for (unsigned i = 0; i < 32; ++i)
      maximum = std::fmax(maximum, std::fabs(input[block * 32 + i]));
    int exponent = 0;
    if (maximum)
      std::frexp(maximum, &exponent);
    int scale = maximum ? exponent - 9 : 0;
    if (scale < -127)
      scale = -127;
    if (output[128 + block] != scale + 127)
      return 1;
    for (unsigned i = 0; i < 32; ++i) {
      unsigned index = block * 32 + i;
      uint8_t expected = encode(input[index], scale);
      if (output[index] != expected) {
        std::printf("MXFP8 mismatch at %u: %02x expected %02x\n", index,
                    output[index], expected);
        return 1;
      }
    }
  }
  std::puts("RVV MXFP8 QUANT PASSED (scalar byte-exact reference)");
  return 0;
}
