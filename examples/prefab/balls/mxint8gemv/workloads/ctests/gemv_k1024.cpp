#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint8gemv.h>
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[1056], b[32 * 1056];
    alignas(64) uint8_t sentinel[272], expected[64];
  } v{};
  for (int k = 0; k < 1056; ++k)
    v.a[k] = k < 512 ? 1 : k < 1024 ? -1 : 127;
  for (int j = 0; j < 32; ++j) {
    int code = j % 7 - 3;
    for (int k = 0; k < 1056; ++k)
      v.b[j * 1056 + k] = k < 512    ? code
                          : k < 1024 ? -code
                          : k < 1040 ? 127
                                     : 126;
    v.expected[j] = uint8_t(24 * code);
  }
  v.expected[32] = 132;
  for (int i = 0; i < 272; ++i)
    v.sentinel[i] = 0x6b;
  for (int i = 48; i < 64; ++i)
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
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output.bytes, fixture.expected, sizeof(output)};
int main() {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mem_alloc(2, 1, 1);
  bb_mvin_strided((uintptr_t)fixture.a, 0, 0, 2, 1, 64, 16, 16, 0);
  bb_mvin_strided((uintptr_t)(fixture.a + 1024), 0, 0, 80, 1, 2, 16, 16, 0);
  for (unsigned g = 0; g < 4; ++g)
    bb_mvin_strided((uintptr_t)(fixture.b + g * 8 * 1056), 1, g, 3, 1, 528, 16,
                    16, 0);
  bb_mvin_strided((uintptr_t)fixture.sentinel, 2, 0, 4, 1, 17, 16, 16, 0);
  bb_mxint8gemv(0, 1, 2, 1024, 2, 80, 3, 4, 0, 1);
  bb_mvout_strided((uintptr_t)output.bytes, 2, 0, 4, 1, 3, 16, 16, 0);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_alloc(15, 1, 1);
  bb_mvin((uintptr_t)output.bytes, 15, 4, 1);
  bb_mem_release(15);
  return 0;
}
