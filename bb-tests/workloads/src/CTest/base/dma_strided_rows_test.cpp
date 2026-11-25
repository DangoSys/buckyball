#include <bbhw/isa/isa.h>
#include <cstdint>

static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) uint8_t input[480], expected[32832];
  } f{};
  for (auto &x : f.expected)
    x = 0x5a;
  for (unsigned row = 0; row < 8; ++row)
    for (unsigned byte = 0; byte < 32; ++byte) {
      uint8_t value = row * 17 + byte;
      f.input[row * 32 + byte] = value;
      f.expected[16 + row * 512 + byte] = value;
    }
  for (unsigned row = 0; row < 7; ++row)
    for (unsigned byte = 0; byte < 32; ++byte) {
      uint8_t value = row * 19 + byte + 3;
      f.input[256 + row * 32 + byte] = value;
      f.expected[4128 + 16 + row * 4096 + byte] = value;
    }
  return f;
}();
alignas(64) static auto output = []() constexpr {
  struct {
    uint8_t bytes[32832];
  } f{};
  for (auto &x : f.bytes)
    x = 0x5a;
  return f;
}();
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output.bytes, fixture.expected, sizeof(output)};
int main() {
  bb_mem_alloc(0, 1, 1);
  bb_mvin((uintptr_t)fixture.input, 0, 16, 1);
  // Each packed eight-column FP32 row occupies two contiguous bank rows.
  bb_mvout_strided((uintptr_t)(output.bytes + 16), 0, 0, 0, 2, 16, 512, 16, 1);
  bb_mvin((uintptr_t)(fixture.input + 256), 0, 14, 1);
  // Seven sparse output rows fit the prepared map at a 4096-byte stride.
  bb_mvout_blocks((uintptr_t)(output.bytes + 4144), 0, 0, 0, 2, 14, 256, 1);
  bb_mem_release(0);
  bb_mem_alloc(0, 1, 1);
  for (unsigned at = 0; at < sizeof(output); at += 256) {
    unsigned bytes = sizeof(output) - at < 256 ? sizeof(output) - at : 256;
    bb_mvin((uintptr_t)(output.bytes + at), 0, bytes / 16, 1);
  }
  bb_mem_release(0);
  return 0;
}
