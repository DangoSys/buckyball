#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/intsoftmax.h>
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) uint32_t input[128], table[4096], expected[136];
  } v{};
  for (unsigned i = 0; i < 4096; ++i) {
    double term = 1, sum = 1;
    for (unsigned k = 1; k <= 24; ++k) {
      term *= -double(i) / (4096 * k);
      sum += term;
    }
    for (unsigned k = 0; k < 4; ++k)
      sum *= sum;
    v.table[i] = uint32_t(sum * 32768 + .5);
  }
  for (auto &x : v.expected)
    x = 0x5a5a5a5a;
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned j = 0; j < 32; ++j) {
      bool visible = j < (16U >> row);
      v.input[row * 32 + j] = visible ? 0 : 0xff800000U;
      v.expected[4 + row * 32 + j] = visible ? 0x3d800000U + (row << 23) : 0;
    }
  return v;
}();
alignas(64) static auto output = []() constexpr {
  struct {
    uint32_t words[136];
  } v{};
  for (auto &x : v.words)
    x = 0x5a5a5a5a;
  return v;
}();
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check"), used, retain)) = {
    output.words, fixture.expected, sizeof(output)};
int main() {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)fixture.input, 0, 32, 1);
  bb_mvin((uintptr_t)fixture.table, 1, 1024, 1);
  bb_intsoftmax(0, 1, 2, 4, 32);
  bb_mvout((uintptr_t)(output.words + 4), 2, 32, 1);
  asm volatile("fence rw,rw" ::: "memory");
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
