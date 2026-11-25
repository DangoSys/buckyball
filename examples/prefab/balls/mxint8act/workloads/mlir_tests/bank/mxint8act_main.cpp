#include <algorithm>
#include <cstdint>

constexpr int K = 32, M = 8, OUT = 48;
constexpr int64_t round_shift(int64_t value, int shift) {
  return __builtin_roundeven(__builtin_ldexp(double(value), -shift));
}
constexpr int64_t lookup(int code, int exponent, const int32_t *table) {
  if (exponent < -24)
    return int64_t(code) * 32768;
  int64_t x = int64_t(code) * (1LL << ((exponent > 4 ? 4 : exponent) + 26));
  if (x >= (16LL << 26))
    return int64_t(code) * 65536;
  if (x <= -(16LL << 26))
    return 0;
  uint64_t at = x + (16LL << 26), i = at >> 19;
  int64_t a = table[i], b = i == 4095 ? 16LL << 26 : table[i + 1];
  return round_shift(a + round_shift((b - a) * (at & ((1 << 19) - 1)), 19),
                     exponent + 10);
}
struct Fixture {
  int32_t table[4096];
  uint8_t input[M * 48], gated[2 * M * 48], expected[2 * M * OUT];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  for (int i = 0; i < 4096; ++i) {
    double x = double(i) / 128 - 16;
    double y =
        x * .5 * (1 + __builtin_erf(x * .7071067811865475244)) * (1 << 26);
    f.table[i] = __builtin_roundeven(y);
  }
  const int aexp[] = {-3, 0, 5, -40, 120, -100, 10, -10};
  const int bexp[] = {-2, 1, -1, 15, 100, -100, 120, -2};
  for (int row = 0; row < M; ++row) {
    for (int i = 0; i < K; ++i) {
      int8_t a = (i * 13 + row) % 127 - 63, b = (i * 19 + row * 3) % 127 - 63;
      if (row == 6)
        a = -(i % 63 + 1);
      if (row == 7)
        b = 0;
      f.input[row * 48 + i] = a;
      f.gated[row * 48 + i] = a;
      f.gated[(M + row) * 48 + i] = b;
    }
    f.input[row * 48 + K] = aexp[row] + 127;
    f.gated[row * 48 + K] = aexp[row] + 127;
    f.gated[(M + row) * 48 + K] = bexp[row] + 127;
    for (int mode = 0; mode < 2; ++mode) {
      int64_t values[K]{};
      uint64_t maximum = 0;
      for (int i = 0; i < K; ++i) {
        values[i] = lookup(int8_t(f.input[row * 48 + i]), aexp[row], f.table);
        if (mode)
          values[i] *= int8_t(f.gated[(M + row) * 48 + i]);
        maximum = std::max(maximum, uint64_t(__builtin_llabs(values[i])));
      }
      int scale = aexp[row] + (mode ? bexp[row] : 0);
      int exponent = maximum ? scale + (63 - __builtin_clzll(maximum)) - 22 : 0;
      exponent = std::clamp(exponent, -127, 127);
      auto *out = &f.expected[(mode * M + row) * OUT];
      for (int i = 0; i < K; ++i) {
        int shift = exponent - scale + 16;
        int64_t code = shift < -6 ? ((values[i] > 0) - (values[i] < 0)) * 127
                                  : round_shift(values[i], shift);
        out[i] = std::clamp<int64_t>(code, -127, 127);
      }
      out[K] = exponent + 127;
    }
  }
  return f;
}();
alignas(64) static uint8_t actual[2 * M * OUT];
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    actual, fixture.expected, sizeof(actual)};
template <int Rank> struct MemRef {
  void *allocated, *aligned;
  int64_t offset, sizes[Rank], strides[Rank];
};
extern "C" void _mlir_ciface_compute(MemRef<2> *, MemRef<2> *, MemRef<1> *,
                                     MemRef<3> *);
int main() {
  MemRef<2> a{
      (void *)fixture.input, (void *)fixture.input, 0, {M, 48}, {48, 1}};
  MemRef<2> g{
      (void *)fixture.gated, (void *)fixture.gated, 0, {2 * M, 48}, {48, 1}};
  MemRef<1> table{(void *)fixture.table, (void *)fixture.table, 0, {4096}, {1}};
  MemRef<3> out{actual, actual, 0, {2, M, OUT}, {M * OUT, OUT, 1}};
  _mlir_ciface_compute(&a, &g, &table, &out);
  return 0;
}
