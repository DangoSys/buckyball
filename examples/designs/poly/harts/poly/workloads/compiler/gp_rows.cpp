#include <cstdint>
#include <cstdio>
#include <kernel.h>

struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct vector {
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern "C" void _mlir_ciface_compute(tensor *, tensor *, vector *, tensor *,
                                     tensor *, tensor *);
struct Fixture {
  float input[15 * 64], scales[15], weights[32], expected[15 * 32];
  int8_t codes[15 * 64];
  uint8_t exponents[15 * 2];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture value{};
  const float powers[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
  for (unsigned i = 0; i < 15 * 64; ++i)
    value.codes[i] = int(i % 127) - 63;
  for (unsigned i = 0; i < 15 * 2; ++i)
    value.exponents[i] = 125 + i % 5;
  for (unsigned row = 0; row < 15; ++row)
    value.scales[row] = (row + 1) * 0.5f;
  for (unsigned column = 0; column < 32; ++column)
    value.weights[column] = (int(column) - 16) * 0.125f;
  for (unsigned i = 0; i < 15 * 32; ++i) {
    float x = (int(i % 17) - 8) * 0.25f;
    value.input[i / 32 * 64 + i % 32] = -123.0f;
    value.input[i / 32 * 64 + i % 32 + 32] = x;
    unsigned row = i / 32, column = i % 32;
    float right = x - value.codes[row * 64 + column + 32] *
                          powers[value.exponents[row * 2 + 1] - 125];
    float left = -123.0f - value.codes[row * 64 + column] *
                               powers[value.exponents[row * 2] - 125];
    value.expected[i] =
        (right * 1.5f + left * 0.25f) * value.scales[0] + value.weights[i % 32];
  }
  return value;
}();
alignas(64) static float output[15 * 32];

int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  auto *data = const_cast<float *>(fixture.input);
  tensor input{data, data, 0, {15, 64}, {64, 1}};
  auto *scale = const_cast<float *>(fixture.scales);
  tensor scales{scale, scale, 0, {15, 1}, {1, 1}};
  auto *weight = const_cast<float *>(fixture.weights);
  vector weights{weight, weight, 0, {32}, {1}};
  tensor result{output, output, 0, {15, 32}, {32, 1}};
  auto *packed = const_cast<int8_t *>(fixture.codes);
  tensor codes{packed, packed, 0, {15, 64}, {64, 1}};
  auto *exponents = const_cast<uint8_t *>(fixture.exponents);
  tensor exponent{exponents, exponents, 0, {15, 2}, {2, 1}};
  _mlir_ciface_compute(&input, &scales, &weights, &codes, &exponent, &result);
  for (unsigned i = 0; i < 15 * 32; ++i)
    if (output[i] != fixture.expected[i]) {
      printf("GP row mismatch at %u: %f != %f\n", i, output[i],
             fixture.expected[i]);
      return 1;
    }
  return 0;
}
