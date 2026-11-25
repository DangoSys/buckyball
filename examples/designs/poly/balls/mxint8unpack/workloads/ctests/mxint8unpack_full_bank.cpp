#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint8unpack.h>
constexpr unsigned B = 128, K = 32 * B, INPUT_BYTES = (33 * B + 15) / 16 * 16;
static constexpr auto fixture = [] {
  struct {
    alignas(64) uint8_t input[INPUT_BYTES];
    uint32_t expected[K + 8];
  } f{};
  for (auto &x : f.input)
    x = 0xd3;
  for (auto &x : f.expected)
    x = 0xa5a5a5a5;
  for (unsigned block = 0; block < B; ++block) {
    unsigned exponent = 2 * block;
    f.input[K + block] = exponent;
    for (unsigned lane = 0; lane < 32; ++lane) {
      const int edge[] = {0, -1, 1, -128, 127};
      int code =
          lane < 5 ? edge[lane] : int((lane * 17 + block * 7) % 256) - 128;
      f.input[32 * block + lane] = uint8_t(code);
      unsigned a = code < 0 ? -code : code, l = 0;
      for (unsigned v = a; v > 1; v >>= 1)
        ++l;
      uint32_t sign = code < 0 ? 0x80000000 : 0;
      unsigned field = exponent + l;
      f.expected[4 + 32 * block + lane] =
          a == 0         ? 0
          : field >= 255 ? sign | 0x7f800000
          : field == 0   ? sign | 0x00400000
                         : sign | (field << 23) | ((a << (23 - l)) & 0x7fffff);
    }
  }
  return f;
}();
alignas(64) static auto output = [] {
  struct {
    uint32_t words[K + 8];
  } x{};
  for (auto &w : x.words)
    w = 0xa5a5a5a5;
  return x;
}();
alignas(64) static uint8_t source_copy[INPUT_BYTES];
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} checks[] __attribute__((section(".bbsim.check"), used, retain)) = {
    {output.words, fixture.expected, sizeof(output)},
    {source_copy, fixture.input, sizeof(source_copy)}};
int main() {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)fixture.input, 1, INPUT_BYTES / 16, 1);
  bb_mxint8unpack(1, 2, B);
  bb_mvout((uintptr_t)(output.words + 4), 2, K / 4, 1);
  bb_mvout((uintptr_t)source_copy, 1, INPUT_BYTES / 16, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
