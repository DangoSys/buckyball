#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/f32add.h>

static constexpr unsigned rows = 128;
struct Fixture {
  uint32_t input[3][rows * 4];
};
alignas(64) static constexpr Fixture fixture = [] {
  constexpr uint32_t values[3][8] = {{0x3f800000, 0xc0200000, 0x7f800000,
                                      0xff800000, 0, 0x80000000, 0x7f7fffff, 1},
                                     {0x40000000, 0x40400000, 0xff800000,
                                      0x7f800000, 0x80000000, 0, 0xff7fffff, 1},
                                     {0xbf800000, 0x3f000000, 0x40a00000,
                                      0xc0a00000, 0, 0x80000000, 0x3f800000,
                                      0x80000001}};
  Fixture f{};
  for (unsigned group = 0; group < 3; ++group)
    for (unsigned i = 0; i < rows * 4; ++i)
      f.input[group][i] = values[group][i % 8];
  return f;
}();
// Independently evaluated ordered FP32 sums, including canonical NaNs.
static constexpr uint32_t expected[8] = {
    0x40000000, 0x3f800000, 0x7fc00000, 0x7fc00000, 0, 0, 0x3f800000, 1};
alignas(64) static uint32_t output[rows * 4];

int main() {
  unsigned accumulator = 1, target = THREAD_FRONTEND_SHARED_VBANK_BASE;
  bb_mem_alloc(0, 1, 3);
  bb_mem_alloc(accumulator, 1, 1);
  bb_mem_alloc(target, 1, 1);
  for (unsigned group = 0; group < 3; ++group)
    bb_mvin_group((uintptr_t)fixture.input[group], 0, group, rows, 1);
  for (unsigned group = 0; group < 3; ++group) {
    bb_f32add(0, accumulator, target, rows, group, group == 0);
    unsigned previous = accumulator;
    accumulator = target;
    target = previous;
  }
  bb_mem_release(0);
  bb_mem_alloc(0, 1, 1);
  bb_f32add(accumulator, target, 0, rows, 0, 1);
  bb_mvout((uintptr_t)output, 0, rows, 1);
  for (unsigned i = 0; i < rows * 4; ++i)
    if (output[i] != expected[i % 8])
      return 1;
  bb_mem_release(0);
  bb_mem_release(accumulator);
  bb_mem_release(target);
  return 0;
}
