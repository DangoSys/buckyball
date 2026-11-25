#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint5gemv.h>
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) uint8_t a[48], b[2048], expected[576];
  } v{};
  for (unsigned k = 0; k < 32; k++)
    v.a[k] = 1;
  v.a[32] = 127;
  for (unsigned j = 0; j < 32; j++) {
    for (unsigned k = 0; k < 32; k++) {
      unsigned bit = 5 * k, at = j * 64 + bit / 8;
      v.b[at] |= uint8_t(1U << (bit % 8));
      if (bit % 8 > 3)
        v.b[at + 1] |= uint8_t(1U >> (8 - bit % 8));
    }
    v.b[j * 64 + 32] = 16;
    v.b[j * 64 + 48] = 127;
  }
  for (unsigned i = 0; i < 16; i++)
    v.expected[i] = v.expected[560 + i] = 0x5a;
  for (unsigned slot = 0; slot < 2; slot++) {
    unsigned at = 16 + slot * 272;
    v.expected[at] = 1;
    v.expected[at + 2] = 0xfc;
    v.expected[at + 3] = 0xff;
    v.expected[at + 4] = 1;
    v.expected[at + 5] = 5;
    v.expected[at + 6] = 28;
    for (unsigned j = 0; j < 32; j++)
      v.expected[at + 20 + j * 8] = 0x20;
  }
  return v;
}();
alignas(64) static auto output = []() constexpr {
  struct {
    uint8_t bytes[576];
  } v{};
  for (auto &x : v.bytes)
    x = 0x5a;
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
  bb_mvin((uintptr_t)fixture.a, 0, 3, 1);
  for (unsigned g = 0; g < 4; g++)
    bb_mvin_strided((uintptr_t)(fixture.b + g * 512), 1, g, 0, 1, 32, 16, 16,
                    0);
  // No clear or DMA initializes C before the first update.
  bb_mxint5gemv(0, 1, 2, 32, 0, 2, 0, 0, 0, 0, 1);
  bb_mxint5gemv(0, 1, 2, 32, 0, 2, 0, 32, 0, 0, 1);
  bb_mvout_strided((uintptr_t)(output.bytes + 16), 2, 0, 0, 1, 17, 16, 16, 0);
  bb_mvout_strided((uintptr_t)(output.bytes + 288), 2, 0, 32, 1, 17, 16, 16, 0);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
