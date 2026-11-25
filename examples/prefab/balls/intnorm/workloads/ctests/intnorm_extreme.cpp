#include <bbhw/isa/isa.h>
#include <cstdint>
#include <isa/intnorm.h>
#include <thread_run.h>

constexpr unsigned K = 512, STRIDE = K + K / 32;
struct Fixture {
  float input[K];
  int32_t coefficients[4 + 2 * K];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  f.coefficients[0] = 42950;
  for (unsigned i = 0; i < K; ++i) {
    f.input[i] = i ? -32768.0f : 32767.0f;
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
  // The positive centered Q16 value exceeds INT32_MAX. Normalization gives
  // approximately sqrt(511) and -1/sqrt(511), with exponents -2 and -11.
  for (unsigned i = 0; i < K; ++i)
    if (int8_t(output[i]) != (i == 0 ? 90 : i < 32 ? 0 : -91))
      return 1;
  for (unsigned i = 0; i < K / 32; ++i)
    if (output[K + i] != (i ? 116 : 125))
      return 2;
  return 0;
}
