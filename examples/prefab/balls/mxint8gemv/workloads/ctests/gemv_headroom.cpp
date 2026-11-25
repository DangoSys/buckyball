#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint8gemv.h>
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[528], b[32 * 528];
    alignas(64) uint8_t expected[336];
  } v{};
  for (int k = 0; k < 528; ++k)
    v.a[k] = k < 512 ? -128 : 127;
  for (int j = 0; j < 32; ++j) {
    for (int k = 0; k < 528; ++k)
      v.b[j * 528 + k] = k < 512 ? -128 : 127;
    v.expected[16 + j * 8 + 7] = 0x20;
    v.expected[272 + j] = 64;
  }
  v.expected[1] = 4;
  v.expected[4] = 1;
  v.expected[304] = 150;
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
  bb_mvin_strided((uintptr_t)fixture.a, 0, 0, 2, 1, 32, 16, 16, 0);
  bb_mvin_strided((uintptr_t)(fixture.a + 512), 0, 0, 40, 1, 1, 16, 16, 0);
  for (unsigned g = 0; g < 4; ++g)
    bb_mvin_strided((uintptr_t)(fixture.b + g * 8 * 528), 1, g, 3, 1, 264, 16,
                    16, 0);
  for (unsigned i = 0; i < 64; ++i)
    bb_mxint8gemv(0, 1, 2, 512, 2, 40, 3, 4, i != 0, 0);
  bb_mvout_strided((uintptr_t)output.bytes, 2, 0, 4, 1, 17, 16, 16, 0);
  for (unsigned i = 0; i < 64; ++i)
    bb_mxint8gemv(0, 1, 2, 512, 2, 40, 3, 4, i != 0, i == 63);
  bb_mvout_strided((uintptr_t)(output.bytes + 272), 2, 0, 4, 1, 3, 16, 16, 0);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_alloc(15, 1, 1);
  bb_mvin((uintptr_t)output.bytes, 15, 21, 1);
  bb_mem_release(15);
  return 0;
}
