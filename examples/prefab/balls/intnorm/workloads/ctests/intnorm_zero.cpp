#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>
#include <isa/intnorm.h>
#include <thread_run.h>

constexpr unsigned K = 512, ROWS = 2, STRIDE = K + K / 32;
struct Fixture {
  float input[ROWS * K];
  int32_t coefficients[4 + 2 * K];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  const int32_t beta[] = {65536, -65536, 32768, -32768};
  f.coefficients[0] = 42950;
  for (unsigned i = 0; i < K; ++i) {
    f.input[i] = 3;
    f.input[K + i] = i & 1 ? -0.0f : 0.0f;
    f.coefficients[4 + i] = 98304;
    f.coefficients[4 + K + i] = i < 32 ? 0 : beta[i % 4];
  }
  return f;
}();
alignas(64) static uint8_t output[ROWS * STRIDE];
static void run(void *) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)fixture.input, 1, ROWS * K / 4, 1);
  bb_mvin((uintptr_t)fixture.coefficients, 2, 1 + K / 2, 1);
  bb_intnorm(1, 2, 3, ROWS, K, 0, 0, 0);
  bb_mvout((uintptr_t)output, 3, ROWS * STRIDE / 16, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
}
int main() {
  bb_thread_run(run, nullptr);
  // Both constant rows normalize to zero. The first block remains zero;
  // the affine bias of every other block is [1, -1, .5, -.5].
  const int8_t codes[] = {64, -64, 32, -32};
  for (unsigned row = 0; row < ROWS; ++row) {
    const uint8_t *p = output + row * STRIDE;
    for (unsigned i = 0; i < K; ++i)
      if (int8_t(p[i]) != (i < 32 ? 0 : codes[i % 4])) {
        printf("Integer norm zero mismatch at %u/%u\n", row, i);
        return 1;
      }
    for (unsigned i = 0; i < K / 32; ++i)
      if (p[K + i] != (i == 0 ? 127 : 121))
        return 2;
  }
  printf("Integer norm zero PASSED\n");
  return 0;
}
