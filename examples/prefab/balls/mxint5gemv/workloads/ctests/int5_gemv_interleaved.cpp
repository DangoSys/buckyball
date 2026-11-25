#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint5gemv.h>
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) uint8_t a[80], b[2048], guard[16384], expected[16384];
  } v{};
  for (unsigned i = 0; i < 32; i++) {
    v.a[i] = 1;
    v.a[32 + i] = 2;
  }
  v.a[64] = 127;
  for (unsigned i = 0; i < 16384; i++)
    v.guard[i] = v.expected[i] = 0x5a;
  for (unsigned j = 0; j < 32; j++) {
    unsigned code = unsigned(int(j % 7) - 3) & 31;
    for (unsigned k = 0; k < 32; k++) {
      unsigned bit = 5 * k, at = j * 64 + bit / 8;
      v.b[at] |= uint8_t(code << (bit % 8));
      if (bit % 8 > 3)
        v.b[at + 1] |= uint8_t(code >> (8 - bit % 8));
    }
    v.b[j * 64 + 32] = 16;
    v.b[j * 64 + 48] = 127;
  }
  for (unsigned slot = 0; slot < 2; slot++) {
    unsigned base = (slot ? 36 : 3) * 16;
    for (unsigned i = 0; i < 272; i++)
      v.expected[base + i] = 0;
    v.expected[base] = 1;
    v.expected[base + 2] = 0xfc;
    v.expected[base + 3] = 0xff;
    v.expected[base + 4] = 1;
    v.expected[base + 5] = 5;
    v.expected[base + 6] = 28;
    for (unsigned j = 0; j < 32; j++) {
      uint64_t value =
          uint64_t((INT64_C(512) * int64_t(slot + 1) * (int(j % 7) - 3)) *
                   (INT64_C(1) << 28));
      for (unsigned byte = 0; byte < 8; byte++)
        v.expected[base + 16 + j * 8 + byte] = uint8_t(value >> (8 * byte));
    }
    // Finalization overwrites exactly three rows; other state bytes survive.
    for (unsigned i = 0; i < 48; i++)
      v.expected[base + i] = 0;
    for (unsigned j = 0; j < 32; j++)
      v.expected[base + j] = uint8_t(24 * (int(j % 7) - 3));
    v.expected[base + 32] = 129;
  }
  return v;
}();
alignas(64) static uint8_t output[16384];
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check"), used, retain)) = {
    output, fixture.expected, sizeof(output)};
int main() {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mem_alloc(2, 1, 1);
  bb_mvin_strided((uintptr_t)fixture.a, 0, 0, 0, 1, 2, 16, 16, 0);
  bb_mvin_strided((uintptr_t)(fixture.a + 64), 0, 0, 2, 1, 1, 16, 16, 0);
  for (unsigned g = 0; g < 4; g++)
    bb_mvin_strided((uintptr_t)(fixture.b + g * 512), 1, g, 0, 1, 32, 16, 16,
                    0);
  bb_mvin((uintptr_t)fixture.guard, 2, 1024, 1);
  bb_mxint5gemv(0, 1, 2, 32, 0, 2, 0, 3, 0, 0, 1);
  bb_mvin_strided((uintptr_t)(fixture.a + 32), 0, 0, 0, 1, 2, 16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 32, 0, 2, 0, 36, 0, 0, 1);
  bb_mxint5gemv(0, 1, 2, 32, 0, 2, 0, 3, 1, 1, 1);
  bb_mvin_strided((uintptr_t)fixture.a, 0, 0, 0, 1, 2, 16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 32, 0, 2, 0, 36, 1, 1, 1);
  bb_mvout((uintptr_t)output, 2, 1024, 1);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
