#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint8unpack.h>
constexpr unsigned K = 4096, PACKED = 4224;
static constexpr auto fixture = [] {
  struct {
    alignas(64) uint8_t input[16][PACKED];
    uint32_t expected[16][K + 8];
  } f{};
  for (unsigned row = 0; row < 16; ++row) {
    for (auto &x : f.expected[row])
      x = 0xa5a5a5a5;
    for (unsigned block = 0; block < 128; ++block) {
      unsigned exponent = 2 * block + (row & 1);
      f.input[row][K + block] = exponent;
      for (unsigned lane = 0; lane < 32; ++lane) {
        int code = int((lane * 17 + (row / 2) * 32) % 256) - 128;
        f.input[row][32 * block + lane] = uint8_t(code);
        unsigned a = code < 0 ? -code : code, l = 0;
        for (unsigned v = a; v > 1; v >>= 1)
          ++l;
        uint32_t sign = code < 0 ? 0x80000000 : 0;
        unsigned field = exponent + l;
        f.expected[row][4 + 32 * block + lane] =
            a == 0         ? 0
            : field >= 255 ? sign | 0x7f800000
            : field == 0   ? sign | 0x00400000
                         : sign | (field << 23) | ((a << (23 - l)) & 0x7fffff);
      }
    }
  }
  return f;
}();
alignas(64) static auto output = [] {
  struct {
    uint32_t words[16][K + 8];
  } x{};
  for (auto &row : x.words)
    for (auto &w : row)
      w = 0xa5a5a5a5;
  return x;
}();
alignas(64) static uint8_t source_copy[16][PACKED];
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} checks[] __attribute__((section(".bbsim.check"), used, retain)) = {
    {output.words, fixture.expected, sizeof(output)},
    {source_copy, fixture.input, sizeof(source_copy)}};
int main() {
  for (unsigned row = 0; row < 16; ++row) {
    bb_mem_alloc(1, 1, 1);
    bb_mem_alloc(2, 1, 1);
    bb_mvin((uintptr_t)fixture.input[row], 1, PACKED / 16, 1);
    bb_mxint8unpack(1, 2, 128);
    bb_mvout((uintptr_t)(output.words[row] + 4), 2, K / 4, 1);
    bb_mvout((uintptr_t)source_copy[row], 1, PACKED / 16, 1);
    bb_mem_release(1);
    bb_mem_release(2);
  }
  return 0;
}
