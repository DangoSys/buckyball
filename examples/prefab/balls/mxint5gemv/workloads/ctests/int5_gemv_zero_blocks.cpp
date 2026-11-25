#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint5gemv.h>

static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) uint8_t a[528], b[32 * 352], expected[336];
  } v{};
  for (unsigned k = 512; k < 528; ++k)
    v.a[k] = 127;
  for (unsigned j = 0; j < 32; ++j) {
    for (unsigned k = 0; k < 512; ++k) {
      const unsigned bit = 5 * k;
      v.b[j * 352 + bit / 8] |= uint8_t(1U << (bit % 8));
    }
    for (unsigned block = 0; block < 16; ++block) {
      v.b[j * 352 + 320 + block] = 17;
      v.b[j * 352 + 336 + block] = 127;
    }
  }
  v.expected[0] = 16;
  v.expected[5] = 5;
  v.expected[6] = 28;
  v.expected[304] = 127;
  for (unsigned i = 320; i < 336; ++i)
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
} check __attribute__((section(".bbsim.check"), used, retain)) = {
    output.bytes, fixture.expected, sizeof(output)};

int main() {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mem_alloc(2, 1, 1);
  bb_mvin_strided((uintptr_t)fixture.a, 0, 0, 2, 1, 32, 16, 16, 0);
  bb_mvin_strided((uintptr_t)(fixture.a + 512), 0, 0, 40, 1, 1, 16, 16, 0);
  for (unsigned group = 0; group < 4; ++group)
    bb_mvin_strided((uintptr_t)(fixture.b + group * 8 * 352), 1, group, 3, 1,
                    176, 16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 512, 2, 40, 3, 4, 0, 0, 0);
  bb_mvout_strided((uintptr_t)output.bytes, 2, 0, 4, 1, 17, 16, 16, 0);
  bb_mxint5gemv(0, 1, 2, 512, 2, 40, 3, 4, 1, 1, 0);
  bb_mvout_strided((uintptr_t)(output.bytes + 272), 2, 0, 4, 1, 3, 16, 16, 0);
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
