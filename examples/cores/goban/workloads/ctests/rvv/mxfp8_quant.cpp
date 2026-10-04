#include "images.h"
#include <bbhw/isa/isa.h>
#include <cmath>
#include <cstdio>
#include <cstring>

alignas(16) static float input[128];
alignas(16) static uint8_t output[144];

// Enumerate finite E4M3 codes independently, choosing nearest, ties to even.
static uint8_t encode(float value, int scale) {
  double magnitude = std::ldexp(std::fabs(double(value)), -scale);
  double distance = magnitude;
  unsigned selected = 0;
  for (unsigned code = 1; code <= 126; ++code) {
    unsigned exponent = code >> 3, fraction = code & 7;
    double candidate = exponent
                           ? std::ldexp(1.0 + fraction / 8.0, int(exponent) - 7)
                           : std::ldexp(double(fraction), -9);
    double error = std::fabs(magnitude - candidate);
    if (error < distance || (error == distance && !(code & 1))) {
      selected = code;
      distance = error;
    }
  }
  return selected | (std::signbit(value) ? 128 : 0);
}

int main() {
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
  for (unsigned bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  alignas(16) uint8_t descriptor[64]{};
  kernel_launch call{images::quant.entry,
                     images::quant.text_bytes,
                     0x80002000,
                     {1 << 16, 2 << 16, 128, 48},
                     0};
  std::memcpy(descriptor, &call, sizeof(call));
  bb_mvin((uintptr_t)input, 2, sizeof(input) / 16, 1);
  bb_mvin((uintptr_t)descriptor, 0, sizeof(descriptor) / 16, 1);
  mvin_kernel(images::quant.bytes, images::quant.size, 0);
  bb_fence();
  run_kernel(0, 0);
  bb_mvout((uintptr_t)descriptor, 0, sizeof(descriptor) / 16, 1);
  bb_mvout((uintptr_t)output, 1, sizeof(output) / 16, 1);
  bb_fence();
  uint32_t status;
  std::memcpy(&status, descriptor + 48, sizeof(status));
  if (status)
    return 1;
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
