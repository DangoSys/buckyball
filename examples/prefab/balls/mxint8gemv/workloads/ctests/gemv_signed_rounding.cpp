#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint8gemv.h>
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[48], b[2][32 * 48];
    alignas(64) uint8_t expected[336];
  } v{};
  v.a[0] = 1;
  v.a[32] = 127;
  for (int t = 0; t < 2; ++t)
    for (int j = 0; j < 32; ++j) {
      int code = j == 31  ? (t ? -1 : 1)
                 : t      ? 0
                 : j == 0 ? 1
                 : j == 1 ? 3
                 : j == 2 ? -1
                 : j == 3 ? -3
                          : 0;
      v.b[t][j * 48] = code;
      v.b[t][j * 48 + 32] = j == 31 ? 160 : 127;
    }
  v.expected[0] = 1;
  v.expected[2] = 33;
  v.expected[4] = 1;
  for (int j = 0; j < 32; ++j) {
    uint64_t x = uint64_t(j == 1    ? 2
                          : j == 3  ? -2
                          : j == 31 ? INT64_C(1) << 32
                                    : 0);
    for (int byte = 0; byte < 8; ++byte)
      v.expected[16 + j * 8 + byte] = x >> (byte * 8);
  }
  v.expected[273] = 64;
  v.expected[275] = uint8_t(-64);
  v.expected[304] = 123;
  for (int i = 320; i < 336; ++i)
    v.expected[i] = 0x5a;
  return v;
}();
alignas(64) static auto output = []() constexpr {
  struct {
    uint8_t bytes[336];
  } v{};
  for (auto &byte : v.bytes)
    byte = 0x5a;
  return v;
}();
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output.bytes, fixture.expected, sizeof(output)};
int main() {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mem_alloc(2, 1, 1);
  bb_mvin_strided((uintptr_t)fixture.a, 0, 0, 2, 1, 2, 16, 16, 0);
  bb_mvin_strided((uintptr_t)(fixture.a + 32), 0, 0, 40, 1, 1, 16, 16, 0);
  for (unsigned t = 0; t < 2; ++t) {
    for (unsigned g = 0; g < 4; ++g)
      bb_mvin_strided((uintptr_t)(fixture.b[t] + g * 8 * 48), 1, g, 3, 1, 24,
                      16, 16, 0);
    bb_mxint8gemv(0, 1, 2, 32, 2, 40, 3, 4, t, t);
    bb_mvout_strided((uintptr_t)(output.bytes + t * 272), 2, 0, 4, 1,
                     t ? 3 : 17, 16, 16, 0);
  }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_alloc(15, 1, 1);
  bb_mvin((uintptr_t)output.bytes, 15, 21, 1);
  bb_mem_release(15);
  return 0;
}
