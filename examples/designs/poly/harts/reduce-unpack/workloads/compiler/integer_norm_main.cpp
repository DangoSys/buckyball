#include <cstdint>
#include <cstdio>
#include <initializer_list>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct vector {
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern "C" void _mlir_ciface_compute(matrix *, vector *, matrix *);
constexpr unsigned K = 1024, PACKED = 1056;
struct Fixture {
  float input[5 + 5 * K];
  int32_t coefficients[4 + 4 + 2 * K];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  const int32_t gamma[] = {65536, 131072, -65536, 32768};
  const int32_t beta[] = {0, 16384, -32768, 65536};
  f.coefficients[4] = 42950;
  for (unsigned i = 0; i < K; ++i) {
    f.coefficients[8 + i] = gamma[i % 4];
    f.coefficients[8 + K + i] = beta[i % 4];
    for (unsigned row = 0; row < 5; ++row)
      f.input[5 + row * K + i] = (i & 1 ? 1.0f : -1.0f) * (row + 1);
  }
  return f;
}();
alignas(64) static uint8_t output[3 + 5 * PACKED];
int main() {
  auto *data = const_cast<float *>(fixture.input);
  auto *params = const_cast<int32_t *>(fixture.coefficients);
  const int8_t codes[] = {-32, 72, 16, 48};
  for (int64_t rows : {1, 2, 5}) {
    matrix input{data, data, 5, {rows, K}, {K, 1}};
    vector coefficients{params, params, 16, {8208}, {1}};
    matrix result{output, output, 3, {rows, PACKED}, {PACKED, 1}};
    _mlir_ciface_compute(&input, &coefficients, &result);
    for (int64_t row = 0; row < rows; ++row) {
      const uint8_t *p = output + 3 + row * PACKED;
      for (unsigned i = 0; i < K; ++i)
        if (int8_t(p[i]) != codes[i % 4]) {
          printf("Integer norm compiler mismatch at %ld/%u\n", row, i);
          return 1;
        }
      for (unsigned i = 0; i < K / 32; ++i)
        if (p[K + i] != 122)
          return 2;
    }
  }
  printf("Integer norm compiler dynamic rows/offsets PASSED\n");
  return 0;
}
