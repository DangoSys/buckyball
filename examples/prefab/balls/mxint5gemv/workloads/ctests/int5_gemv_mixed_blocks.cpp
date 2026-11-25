#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint5gemv.h>

static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[1056];
    alignas(64) uint8_t b[32 * 704], expected[64];
  } v{};
  int64_t sums[32]{};
  for (unsigned k = 0; k < 1024; ++k)
    v.a[k] = int8_t(int((k * 17) % 255) - 127);
  for (unsigned block = 0; block < 32; ++block)
    v.a[1024 + block] = int8_t(127 + int(block % 5) - 2);
  for (unsigned j = 0; j < 32; ++j) {
    for (unsigned block = 0; block < 32; ++block) {
      int64_t dot = 0;
      const unsigned factor = (block * 13 + j * 3) % 137 + 1;
      const int ea = int(block % 5) - 2;
      const int eb = int((block * 3 + j) % 5) - 2;
      for (unsigned i = 0; i < 32; ++i) {
        const unsigned k = block * 32 + i;
        const int code = int((k * 3 + j * 5 + block * 7) % 31) - 15;
        const unsigned bit = 5 * k;
        const unsigned encoded = unsigned(code) & 31;
        v.b[j * 704 + bit / 8] |= uint8_t(encoded << (bit % 8));
        if (bit % 8 > 3)
          v.b[j * 704 + bit / 8 + 1] |= uint8_t(encoded >> (8 - bit % 8));
        dot += int64_t(v.a[k]) * code;
      }
      v.b[j * 704 + 640 + block] = uint8_t(factor);
      v.b[j * 704 + 672 + block] = uint8_t(127 + eb);
      // Exact sum at one fixed Q28 scale; no incremental rescaling reference.
      sums[j] += dot * factor * (INT64_C(1) << (28 + ea + eb - 4));
    }
  }
  uint64_t maximum = 0;
  for (auto value : sums) {
    const uint64_t magnitude = uint64_t(value < 0 ? -value : value);
    if (magnitude > maximum)
      maximum = magnitude;
  }
  unsigned highest = 0;
  for (auto value = maximum; value > 1; value >>= 1)
    ++highest;
  const int exponent = int(highest) - 34;
  const unsigned shift = unsigned(exponent + 28);
  const uint64_t half = UINT64_C(1) << (shift - 1);
  for (unsigned j = 0; j < 32; ++j) {
    const uint64_t magnitude = uint64_t(sums[j] < 0 ? -sums[j] : sums[j]);
    const uint64_t quotient = magnitude >> shift;
    const uint64_t remainder = magnitude & ((UINT64_C(1) << shift) - 1);
    const uint64_t rounded =
        quotient + (remainder > half || (remainder == half && (quotient & 1)));
    const int bounded = rounded > 127 ? 127 : int(rounded);
    v.expected[j] = uint8_t(sums[j] < 0 ? -bounded : bounded);
  }
  v.expected[32] = uint8_t(exponent + 127);
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
