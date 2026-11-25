#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>

extern "C" float compute(float);

int main() {
  const uint32_t inputs[] = {0xff7fffff, 0xff800000, 0xc2d00001, 0xc2d00000,
                             0xc2cfffff, 0xc2ce0000, 0xc2b00000, 0xbf800000,
                             0x80000000, 0x00000000, 0x3f800000, 0x42b00000,
                             0x7f800000, 0x7fc00000};
  for (uint32_t bits : inputs) {
    const float input = std::bit_cast<float>(bits);
    const float expected = expf(input);
    const float actual = compute(input);
    if (std::isnan(expected) ? !std::isnan(actual)
                             : std::bit_cast<uint32_t>(actual) !=
                                   std::bit_cast<uint32_t>(expected)) {
      printf("FAIL: exp input %08x expected %08x actual %08x\n", bits,
             std::bit_cast<uint32_t>(expected),
             std::bit_cast<uint32_t>(actual));
      return 1;
    }
  }
  printf("PASS: FP32 exp underflow, subnormal, finite and special values\n");
  return 0;
}
