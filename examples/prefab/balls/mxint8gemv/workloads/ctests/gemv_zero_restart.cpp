#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint8gemv.h>
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[2][48], b[2][32 * 48];
    alignas(64) uint8_t sentinel[272], expected[384];
  } v{};
  v.a[0][0] = 1;
  v.a[1][0] = -1;
  v.a[0][32] = v.a[1][32] = 127;
  for (int t = 0; t < 2; ++t)
    for (int j = 0; j < 32; ++j) {
      v.b[t][j * 48] = 1;
      v.b[t][j * 48 + 32] = t ? 67 : 127;
    }
  v.expected[0] = 2;
  for (int j = 0; j < 32; ++j)
    v.expected[272 + j] = v.expected[320 + j] = 64;
  v.expected[304] = v.expected[352] = 61;
  for (int i = 0; i < 272; ++i)
    v.sentinel[i] = 0x5a;
  for (int i = 368; i < 384; ++i)
    v.expected[i] = 0x5a;
  return v;
}();
alignas(64) static auto output = []() constexpr {
  struct {
    uint8_t bytes[384];
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
  bb_mvin_strided((uintptr_t)(fixture.a[0] + 32), 0, 0, 40, 1, 1, 16, 16, 0);
  for (unsigned g = 0; g < 4; ++g)
    bb_mvin_strided((uintptr_t)(fixture.b[0] + g * 8 * 48), 1, g, 3, 1, 24, 16,
                    16, 0);
  for (unsigned t = 0; t < 2; ++t) {
    bb_mvin_strided((uintptr_t)fixture.a[t], 0, 0, 2, 1, 2, 16, 16, 0);
    bb_mxint8gemv(0, 1, 2, 32, 2, 40, 3, 4, t, 0);
  }
  bb_mvout_strided((uintptr_t)output.bytes, 2, 0, 4, 1, 17, 16, 16, 0);
  bb_mvin_strided((uintptr_t)fixture.a[0], 0, 0, 2, 1, 2, 16, 16, 0);
  for (unsigned g = 0; g < 4; ++g)
    bb_mvin_strided((uintptr_t)(fixture.b[1] + g * 8 * 48), 1, g, 3, 1, 24, 16,
                    16, 0);
  bb_mxint8gemv(0, 1, 2, 32, 2, 40, 3, 4, 1, 1);
  bb_mvout_strided((uintptr_t)(output.bytes + 272), 2, 0, 4, 1, 3, 16, 16, 0);
  bb_mvin_strided((uintptr_t)fixture.sentinel, 2, 0, 4, 1, 17, 16, 16, 0);
  bb_mxint8gemv(0, 1, 2, 32, 2, 40, 3, 4, 0, 1);
  bb_mvout_strided((uintptr_t)(output.bytes + 320), 2, 0, 4, 1, 3, 16, 16, 0);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_alloc(15, 1, 1);
  bb_mvin((uintptr_t)output.bytes, 15, 24, 1);
  bb_mem_release(15);
  return 0;
}
