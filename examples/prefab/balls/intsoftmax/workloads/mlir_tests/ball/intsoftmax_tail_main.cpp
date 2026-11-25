#include <CRunnerUtils.h>
#include <cstdint>
extern "C" void _mlir_ciface_compute(StridedMemRefType<int32_t, 1> *,
                                     StridedMemRefType<int32_t, 1> *,
                                     StridedMemRefType<int32_t, 1> *);
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
    alignas(64) uint32_t input[16], table[4096], expected[24];
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
  constexpr int source[7] = {1, 3, 5, -1, -3, -5, 2};
  constexpr int rounded[7] = {0, 2, 2, 0, -2, -2, 1};
  for (auto &x : v.expected)
    x = 0x5a5a5a5a;
  for (unsigned row = 0; row < 2; ++row) {
    int sign = row == 0 ? 1 : -1;
    uint64_t sum = 0;
    for (unsigned j = 0; j < 7; ++j) {
      v.input[row * 8 + j] = encoded(sign * source[j], 9);
      sum += v.table[2 - sign * rounded[j]];
    }
    for (unsigned j = 0; j < 7; ++j) {
      uint32_t w = v.table[2 - sign * rounded[j]];
      v.expected[4 + row * 8 + j] =
          encoded((uint64_t(w) * 32768 + sum / 2) / sum, 15);
    }
    v.expected[4 + row * 8 + 7] = 0;
  }
  return v;
}();
alignas(64) static auto output = []() constexpr {
  struct {
    uint32_t words[24];
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
  auto *ip = reinterpret_cast<int32_t *>(const_cast<uint32_t *>(fixture.input));
  auto *tp = reinterpret_cast<int32_t *>(const_cast<uint32_t *>(fixture.table));
  auto *op = reinterpret_cast<int32_t *>(output.words + 4);
  StridedMemRefType<int32_t, 1> input{ip, ip, 0, {16}, {1}};
  StridedMemRefType<int32_t, 1> table{tp, tp, 0, {4096}, {1}};
  StridedMemRefType<int32_t, 1> out{op, op, 0, {16}, {1}};
  _mlir_ciface_compute(&input, &table, &out);
  return 0;
}
