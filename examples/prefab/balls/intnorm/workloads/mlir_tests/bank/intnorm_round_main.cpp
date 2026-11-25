#include <cstdint>
#include <cstdio>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern "C" void _mlir_ciface_normalize(matrix *, matrix *, matrix *);
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
  auto *x = const_cast<float *>(fixture.input);
  auto *p = const_cast<int32_t *>(fixture.coefficients);
  matrix input{x, x, 0, {128, 4}, {4, 1}};
  matrix coefficients{p, p, 0, {257, 4}, {4, 1}};
  matrix result{output, output, 0, {33, 16}, {16, 1}};
  _mlir_ciface_normalize(&input, &coefficients, &result);
  const int8_t codes[] = {-40, -40, 0, 0, 40, 40, 79, -79};
  for (unsigned i = 0; i < K; ++i)
    if (int8_t(output[i]) != codes[i % 8])
      return 1;
  for (unsigned i = 0; i < K / 32; ++i)
    if (output[K + i] != 115)
      return 2;
  printf("Integer norm bank MLIR PASSED\n");
  return 0;
}
