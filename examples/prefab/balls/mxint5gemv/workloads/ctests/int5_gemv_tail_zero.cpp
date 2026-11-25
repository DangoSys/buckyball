#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint5gemv.h>

static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[80];
    alignas(64) uint8_t b[32 * 64];
    alignas(64) uint8_t expected[400];
  } v{};
  for (unsigned k = 0; k < 32; ++k)
    v.a[k] = -128;
  v.a[64] = 127;
  for (unsigned j = 0; j < 32; ++j) {
    const int code = int(j % 31) - 15;
    for (unsigned k = 0; k < 32; ++k) {
      const unsigned bit = 5 * k;
      const unsigned encoded = unsigned(code) & 31;
      v.b[j * 64 + bit / 8] |= uint8_t(encoded << (bit % 8));
      if (bit % 8 > 3)
        v.b[j * 64 + bit / 8 + 1] |= uint8_t(encoded >> (8 - bit % 8));
    }
    v.b[j * 64 + 32] = 137;
    v.b[j * 64 + 48] = 127;
    const uint64_t value =
        uint64_t(int64_t(-4096 * code * 137) * (INT64_C(1) << 28));
    for (unsigned byte = 0; byte < 8; ++byte)
      v.expected[16 + j * 8 + byte] = uint8_t(value >> (8 * byte));
    const int numerator = -137 * code;
    const unsigned magnitude = unsigned(numerator < 0 ? -numerator : numerator);
    const unsigned remainder = magnitude % 32;
    const unsigned rounded =
        magnitude / 32 +
        (remainder > 16 || (remainder == 16 && ((magnitude / 32) & 1)));
    v.expected[272 + j] = uint8_t(numerator < 0 ? -int(rounded) : int(rounded));
  }
  v.expected[0] = 1;
  v.expected[2] = 0xfc;
  v.expected[3] = 0xff;
  v.expected[4] = 1;
  v.expected[5] = 5;
  v.expected[6] = 28;
  v.expected[304] = 140;
  v.expected[368] = 127;
  for (unsigned i = 320; i < 336; ++i)
    v.expected[i] = 0x5a;
  for (unsigned i = 384; i < 400; ++i)
    v.expected[i] = 0x5a;
  return v;
}();

alignas(64) static auto output = []() constexpr {
  struct {
    uint8_t bytes[400];
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
  bb_mvin_strided((uintptr_t)fixture.a, 0, 0, 2, 1, 2, 16, 16, 0);
  bb_mvin_strided((uintptr_t)(fixture.a + 64), 0, 0, 40, 1, 1, 16, 16, 0);
  for (unsigned group = 0; group < 4; ++group)
    bb_mvin_strided((uintptr_t)(fixture.b + group * 8 * 64), 1, group, 3, 1, 32,
                    16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 32, 2, 40, 3, 4, 0, 0, 0);
  bb_mvout_strided((uintptr_t)output.bytes, 2, 0, 4, 1, 17, 16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 32, 2, 40, 3, 4, 0, 1, 0);
  bb_mvout_strided((uintptr_t)(output.bytes + 272), 2, 0, 4, 1, 3, 16, 16, 0);
  bb_mvin_strided((uintptr_t)(fixture.a + 32), 0, 0, 2, 1, 2, 16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 32, 2, 40, 3, 4, 0, 1, 0);
  bb_mvout_strided((uintptr_t)(output.bytes + 336), 2, 0, 4, 1, 3, 16, 16, 0);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
