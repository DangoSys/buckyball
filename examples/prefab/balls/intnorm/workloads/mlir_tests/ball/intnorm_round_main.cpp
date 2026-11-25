#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>
extern "C" void _mlir_ciface_normalize();
constexpr unsigned K = 512;
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
alignas(64) static uint8_t output[K + K / 32];
int main() {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)fixture.input, 1, K / 4, 1);
  bb_mvin((uintptr_t)fixture.coefficients, 2, 1 + K / 2, 1);
  _mlir_ciface_normalize();
  bb_mvout((uintptr_t)output, 3, (K + K / 32) / 16, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  const int8_t codes[] = {-40, -40, 0, 0, 40, 40, 79, -79};
  for (unsigned i = 0; i < K; ++i)
    if (int8_t(output[i]) != codes[i % 8])
      return 1;
  for (unsigned i = 0; i < K / 32; ++i)
    if (output[K + i] != 115)
      return 2;
  printf("Integer norm Ball MLIR PASSED\n");
  return 0;
}
