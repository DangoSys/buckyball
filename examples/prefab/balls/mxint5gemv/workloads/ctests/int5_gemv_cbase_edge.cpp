#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint5gemv.h>
static volatile uint64_t cbase = 1007;
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) uint8_t a[48], b[2048], guard[288], expected[304];
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
  for (auto &x : v.guard)
    x = 0x5a;
  for (auto &x : v.expected)
    x = 0x5a;
  for (unsigned i = 16; i < 288; i++)
    v.expected[i] = 0;
  v.expected[16] = 1;
  v.expected[18] = 0xfc;
  v.expected[19] = 0xff;
  v.expected[20] = 1;
  v.expected[21] = 5;
  v.expected[22] = 28;
  for (unsigned j = 0; j < 32; j++)
    v.expected[36 + j * 8] = 0x20;
  return v;
}();
alignas(64) static auto output = []() constexpr {
  struct {
    uint8_t bytes[304];
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
  bb_mvin_strided((uintptr_t)fixture.guard, 2, 0, 1006, 1, 18, 16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 32, 0, 2, 0, cbase, 0, 0, 1);
  bb_mvout_strided((uintptr_t)output.bytes, 2, 0, 1006, 1, 18, 16, 16, 0);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
