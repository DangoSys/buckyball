#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>
#include <isa/intnorm.h>
#include <thread_run.h>

constexpr unsigned K = 1024, STRIDE = K + K / 32;
struct Fixture {
  float input[K];
  int32_t coefficients[4 + 2 * K];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  const int32_t gamma[] = {65536, 131072, -65536, 32768};
  const int32_t beta[] = {0, 16384, -32768, 65536};
  f.coefficients[0] = 42950;
  for (unsigned i = 0; i < K; ++i) {
    f.input[i] = i & 1 ? 1.0f : -1.0f;
    f.coefficients[4 + i] = gamma[i % 4];
    f.coefficients[4 + K + i] = beta[i % 4];
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
  const int8_t codes[] = {-32, 72, 16, 48};
  for (unsigned i = 0; i < K; ++i)
    if (int8_t(output[i]) != codes[i % 4])
      return 1;
  for (unsigned i = 0; i < K / 32; ++i)
    if (output[K + i] != 122)
      return 2;
  printf("Integer norm wide PASSED\n");
  return 0;
}
