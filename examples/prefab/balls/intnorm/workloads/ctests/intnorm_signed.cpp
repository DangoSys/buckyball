#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>
#include <isa/intnorm.h>
#include <thread_run.h>

constexpr unsigned K = 512, ROWS = 2, IN = 2, COEFF = 1, OUT = 3;
constexpr unsigned STRIDE = K + K / 32, BYTES = OUT * 16 + ROWS * STRIDE;
struct Fixture {
  float input[IN * 4 + ROWS * K];
  int32_t coefficients[COEFF * 4 + 4 + 2 * K];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  const int32_t gamma[] = {65536, 131072, -65536, 32768};
  const int32_t beta[] = {0, 16384, -32768, 65536};
  auto *p = f.coefficients + COEFF * 4;
  p[0] = 42950;
  for (unsigned i = 0; i < K; ++i) {
    p[4 + i] = gamma[i % 4];
    p[4 + K + i] = beta[i % 4];
    for (unsigned row = 0; row < ROWS; ++row)
      f.input[IN * 4 + row * K + i] = (i & 1 ? 1.0f : -1.0f) * (row + 1);
  }
  return f;
}();
alignas(64) static uint8_t output[BYTES];
static void run(void *) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)fixture.input, 1, IN + ROWS * K / 4, 1);
  bb_mvin((uintptr_t)fixture.coefficients, 2, COEFF + 1 + K / 2, 1);
  bb_intnorm(1, 2, 3, ROWS, K, IN, COEFF, OUT);
  bb_mvout((uintptr_t)output, 3, BYTES / 16, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
}
int main() {
  bb_thread_run(run, nullptr);
  // The integer roots are exactly 65536 and 131072. Affine Q16 values
  // are [-1, 2.25, .5, 1.5], packed at exponent -5 in both rows.
  const int8_t codes[] = {-32, 72, 16, 48};
  for (unsigned row = 0; row < ROWS; ++row) {
    const uint8_t *p = output + OUT * 16 + row * STRIDE;
    for (unsigned i = 0; i < K; ++i)
      if (int8_t(p[i]) != codes[i % 4]) {
        printf("Integer norm signed mismatch at %u/%u\n", row, i);
        return 2;
      }
    for (unsigned i = 0; i < K / 32; ++i)
      if (p[K + i] != 122)
        return 3;
  }
  printf("Integer norm signed PASSED\n");
  return 0;
}
