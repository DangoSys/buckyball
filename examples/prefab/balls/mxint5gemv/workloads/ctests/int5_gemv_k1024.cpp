#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint5gemv.h>

static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[1056];
    alignas(64) uint8_t b[32 * 704];
    alignas(64) uint8_t expected[64];
  } v{};
  for (unsigned k = 0; k < 1024; ++k)
    v.a[k] = 1;
  for (unsigned block = 0; block < 32; ++block)
    v.a[1024 + block] = block < 16 ? 127 : 128;
  for (unsigned j = 0; j < 32; ++j) {
    const int code = int(j % 7) - 3;
    const int scale = int(j % 3) - 1;
    for (unsigned k = 0; k < 1024; ++k) {
      const unsigned bit = 5 * k;
      const unsigned encoded = unsigned(code) & 31;
      v.b[j * 704 + bit / 8] |= uint8_t(encoded << (bit % 8));
      if (bit % 8 > 3)
        v.b[j * 704 + bit / 8 + 1] |= uint8_t(encoded >> (8 - bit % 8));
    }
    for (unsigned block = 0; block < 32; ++block) {
      v.b[j * 704 + 640 + block] = block < 16 ? 17 : 137;
      v.b[j * 704 + 672 + block] = uint8_t(127 + scale + (block >= 16));
    }
    const int numerator = 565 * code;
    const unsigned divisor = scale < 0 ? 64 : scale ? 16 : 32;
    const unsigned magnitude = unsigned(numerator < 0 ? -numerator : numerator);
    const unsigned quotient = magnitude / divisor;
    const unsigned remainder = magnitude % divisor;
    const unsigned rounded =
        quotient + (remainder > divisor / 2 ||
                    (remainder == divisor / 2 && (quotient & 1)));
    v.expected[j] = uint8_t(numerator < 0 ? -int(rounded) : int(rounded));
  }
  v.expected[32] = 137;
  for (unsigned i = 48; i < 64; ++i)
    v.expected[i] = 0x5a;
  return v;
}();

alignas(64) static auto output = []() constexpr {
  struct {
    uint8_t bytes[64];
  } v{};
  for (auto &byte : v.bytes)
    byte = 0x5a;
  return v;
}();

static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check"), used, retain)) = {
    output.bytes, fixture.expected, sizeof(output)};

int main() {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mem_alloc(2, 1, 1);
  bb_mvin_strided((uintptr_t)fixture.a, 0, 0, 2, 1, 64, 16, 16, 0);
  bb_mvin_strided((uintptr_t)(fixture.a + 1024), 0, 0, 80, 1, 2, 16, 16, 0);
  for (unsigned group = 0; group < 4; ++group)
    bb_mvin_strided((uintptr_t)(fixture.b + group * 8 * 704), 1, group, 3, 1,
                    352, 16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 1024, 2, 80, 3, 4, 0, 1, 0);
  bb_mvout_strided((uintptr_t)output.bytes, 2, 0, 4, 1, 3, 16, 16, 0);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
