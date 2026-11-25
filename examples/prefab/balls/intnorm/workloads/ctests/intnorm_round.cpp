#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>
#include <isa/intnorm.h>
#include <thread_run.h>

constexpr unsigned K = 512, STRIDE = K + K / 32;
struct Fixture {
  float input[K];
  int32_t coefficients[4 + 2 * K];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  const float values[] = {-2.5f, -1.5f, -.5f, .5f, 1.5f, 2.5f, 3.5f, -3.5f};
  f.coefficients[0] = 42950;
  for (unsigned i = 0; i < K; ++i) {
    f.input[i] = values[i % 8] / 65536;
    f.coefficients[4 + i] = 65536;
  }
  return f;
}();
alignas(64) static uint8_t output[STRIDE];
static void run(void *) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)fixture.input, 1, K / 4, 1);
  bb_mvin((uintptr_t)fixture.coefficients, 2, 1 + K / 2, 1);
  bb_intnorm(1, 2, 3, 1, K, 0, 0, 0);
  bb_mvout((uintptr_t)output, 3, STRIDE / 16, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
}
int main() {
  bb_thread_run(run, nullptr);
  // Q16 RNE gives [-2,-2,0,0,2,2,4,-4], mean 0 and variance 6.
  // sqrt_floor(6+42950)=207, so the normalized Q16 magnitudes are
  // 633 and 1266. Their MXINT8 exponent is -12.
  const int8_t codes[] = {-40, -40, 0, 0, 40, 40, 79, -79};
  for (unsigned i = 0; i < K; ++i)
    if (int8_t(output[i]) != codes[i % 8]) {
      printf("Integer norm rounding mismatch at %u\n", i);
      return 1;
    }
  for (unsigned i = 0; i < K / 32; ++i)
    if (output[K + i] != 115)
      return 2;
  printf("Integer norm rounding PASSED\n");
  return 0;
}
