#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <kernel.h>

constexpr int width = 516;
constexpr std::array<float, 4> scales{0.5f, 2.0f, 1.25f, 0.75f};
struct Data {
  std::array<float, 4 * width> input, expected;
  std::array<float, width> bias;
};
constexpr Data data = [] {
  Data result{};
  for (int column = 0; column < width; ++column) {
    result.bias[column] = (column % 11 - 5) * 0.25f;
    for (int row = 0; row < 4; ++row) {
      uint32_t bits = UINT32_C(0x3f000001) + column * 7919 + row * 313;
      if (column & 1)
        bits |= UINT32_C(0x80000000);
      float value = std::bit_cast<float>(bits);
      result.input[row * width + column] = value;
      float sum = value + result.bias[column];
      float scaled = sum * scales[row];
      float added = scaled + value;
      float again = added * scales[row];
      result.expected[row * width + column] = again - result.bias[column];
    }
  }
  return result;
}();
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern "C" void _mlir_ciface_compute(matrix *, matrix *, matrix *, matrix *);
static float output[4 * width];
int main() {
  kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  auto input = const_cast<float *>(data.input.data());
  auto bias = const_cast<float *>(data.bias.data());
  auto scale = const_cast<float *>(scales.data());
  matrix a{input, input, 0, {4, width}, {width, 1}};
  matrix b{bias, bias, 0, {1, width}, {width, 1}};
  matrix s{scale, scale, 0, {4, 1}, {1, 1}};
  matrix out{output, output, 0, {4, width}, {width, 1}};
  _mlir_ciface_compute(&a, &b, &s, &out);
  if (std::memcmp(output, data.expected.data(), sizeof(output))) {
    std::puts("Compiled RVV row partitions mismatch");
    return 1;
  }
  std::puts("Compiled RVV rows: four partitions, vector tail and two-stage "
            "chain passed");
  return 0;
}
