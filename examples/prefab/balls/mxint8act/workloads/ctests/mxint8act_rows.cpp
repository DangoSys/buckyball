#include <algorithm>
#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/mxint8act.h>
#include <thread_run.h>

constexpr int K = 1024, M = 4;
constexpr int OUT = (K + K / 32 + 15) / 16 * 16, IN = OUT, GIN = 2 * IN;
constexpr int64_t round_shift(int64_t value, int shift) {
  return __builtin_roundeven(__builtin_ldexp(double(value), -shift));
}
constexpr int64_t lookup(int code, int exponent, const int32_t *table) {
  if (exponent < -24)
    return int64_t(code) * 32768;
  int64_t x = int64_t(code) * (1LL << ((exponent > 4 ? 4 : exponent) + 26));
  if (__builtin_llabs(x) >= (16LL << 26))
    return x > 0 ? int64_t(code) * 65536 : 0;
  uint64_t at = x + (16LL << 26), i = at >> 19;
  int64_t a = table[i], b = i == 4095 ? 16LL << 26 : table[i + 1];
  return round_shift(a + round_shift((b - a) * (at & ((1 << 19) - 1)), 19),
                     exponent + 10);
}
struct Fixture {
  int32_t table[4096];
  uint8_t input[M * IN], gated[M * GIN], expected[2 * M * OUT];
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
      f.input[row * IN + i] = f.gated[row * IN + i] = a;
      f.gated[(M + row) * IN + i] = b;
    }
    for (int block = 0; block < K / 32; ++block) {
      int ae = aexp[row] + block % 3, be = bexp[row] - block % 3;
      f.input[row * IN + K + block] = f.gated[row * IN + K + block] = ae + 127;
      f.gated[(M + row) * IN + K + block] = be + 127;
      for (int mode = 0; mode < 2; ++mode) {
        int64_t values[32]{};
        uint64_t maximum = 0;
        for (int i = 0; i < 32; ++i) {
          values[i] =
              lookup(int8_t(f.input[row * IN + block * 32 + i]), ae, f.table);
          if (mode)
            values[i] *= int8_t(f.gated[(M + row) * IN + block * 32 + i]);
          maximum = std::max(maximum, uint64_t(__builtin_llabs(values[i])));
        }
        int scale = ae + (mode ? be : 0);
        int exponent =
            maximum ? scale + (63 - __builtin_clzll(maximum)) - 22 : 0;
        exponent = std::clamp(exponent, -127, 127);
        auto *out = &f.expected[(mode * M + row) * OUT];
        for (int i = 0; i < 32; ++i) {
          int shift = exponent - scale + 16;
          int64_t code = shift < -6 ? ((values[i] > 0) - (values[i] < 0)) * 127
                                    : round_shift(values[i], shift);
          out[block * 32 + i] = std::clamp<int64_t>(code, -127, 127);
        }
        out[K + block] = exponent + 127;
      }
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
static void run(void *) {
  for (int bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)fixture.table, 1, sizeof(fixture.table) / 16, 1);
  for (int mode = 0; mode < 2; ++mode) {
    bb_mvin((uintptr_t)(mode ? fixture.gated : fixture.input), 0,
            M * (mode ? GIN : IN) / 16, 1);
    bb_mxint8act(0, 1, 2, M, K, 0, 0, 0, mode);
    bb_mvout((uintptr_t)(actual + mode * M * OUT), 2, M * OUT / 16, 1);
  }
  for (int bank = 0; bank < 3; ++bank)
    bb_mem_release(bank);
  bb_mem_alloc(15, 1, 1);
  bb_mvin((uintptr_t)actual, 15, sizeof(actual) / 16, 1);
  bb_mem_release(15);
}

int main() {
  bb_thread_run(run, nullptr);
  return 0;
}
