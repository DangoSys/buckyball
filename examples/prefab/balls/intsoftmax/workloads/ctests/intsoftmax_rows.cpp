#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/intsoftmax.h>
static constexpr uint32_t encoded(int32_t x, unsigned fraction) {
  if (!x)
    return 0;
  uint32_t a = x < 0 ? uint32_t(-x) : uint32_t(x);
  unsigned high = 0;
  for (uint32_t t = a; t >>= 1;)
    ++high;
  return (x < 0 ? 0x80000000U : 0) | ((127 + high - fraction) << 23) |
         ((a << (23 - high)) & 0x7fffff);
}
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) uint32_t input[1024], table[4096], expected[1032];
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
  uint64_t sum = 0;
  for (unsigned j = 0; j < 128; ++j)
    sum += v.table[127 - j];
  for (auto &x : v.expected)
    x = 0x5a5a5a5a;
  for (unsigned row = 0; row < 8; ++row)
    for (unsigned j = 0; j < 128; ++j) {
      int offset = row & 1 ? -512 : 4096;
      v.input[row * 128 + j] = encoded(int(j) + offset - 64, 8);
      v.expected[4 + row * 128 + j] =
          encoded((uint64_t(v.table[127 - j]) * 32768 + sum / 2) / sum, 15);
    }
  return v;
}();
alignas(64) static auto output = []() constexpr {
  struct {
    uint32_t words[1032];
  } v{};
  for (auto &x : v.words)
    x = 0x5a5a5a5a;
  return v;
}();
alignas(64) static uint32_t input_after[1024], table_after[4096];
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} checks[] __attribute__((section(".bbsim.check"), used, retain)) = {
    {output.words, fixture.expected, sizeof(output)},
    {input_after, fixture.input, sizeof(input_after)},
    {table_after, fixture.table, sizeof(table_after)},
};
int main() {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)fixture.input, 0, 256, 1);
  bb_mvin((uintptr_t)fixture.table, 1, 1024, 1);
  bb_intsoftmax(0, 1, 2, 8, 128);
  bb_mvout((uintptr_t)(output.words + 4), 2, 256, 1);
  bb_mvout((uintptr_t)input_after, 0, 256, 1);
  bb_mvout((uintptr_t)table_after, 1, 1024, 1);
  asm volatile("fence rw,rw" ::: "memory");
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
