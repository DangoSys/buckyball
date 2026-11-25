#include <kernel.h>
#include <stdint.h>

struct matrix {
  const void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct layout {
  const void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" void _mlir_ciface_compute(matrix *, layout *, matrix *);
struct Fixture {
  float a[12][3072];
  int8_t weights[1][384][8][16];
  float expected[12][16];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  for (int row = 0; row < 12; ++row)
    for (int k = 0; k < 3072; ++k)
      f.a[row][k] = float((row + k) % 7 - 3);
  for (int k = 0; k < 3072; ++k)
    for (int col = 0; col < 16; ++col)
      f.weights[0][k % 384][k / 384][col] = (col * 3 + k) % 5 - 2;
  for (int row = 0; row < 12; ++row)
    for (int col = 0; col < 16; ++col) {
      // The input product repeats every 35 reduction elements.
      int period = 0, tail = 0;
      for (int k = 0; k < 35; ++k) {
        int value = ((row + k) % 7 - 3) * ((col * 3 + k) % 5 - 2);
        period += value;
        if (k < 3072 % 35)
          tail += value;
      }
      f.expected[row][col] = float(period * (3072 / 35) + tail);
    }
  return f;
}();
alignas(64) static float output[12][16];
__attribute__((used, section(".bbsim.check_output"),
               aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {output, fixture.expected, sizeof(output)};

int main() {
  kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  matrix a{fixture.a, fixture.a, 0, {12, 3072}, {3072, 1}};
  layout w{fixture.weights, fixture.weights, 0, {1, 384, 128}, {49152, 128, 1}};
  matrix out{output, output, 0, {12, 16}, {16, 1}};
  _mlir_ciface_compute(&a, &w, &out);
  auto *exit = reinterpret_cast<volatile uint32_t *>(uintptr_t(0x60000000));
  *exit = UINT32_C(0x42424f55);
  for (unsigned offset = 0; offset < sizeof(output); offset += 4) {
    uint32_t word;
    __builtin_memcpy(&word, reinterpret_cast<const char *>(output) + offset, 4);
    *exit = word;
  }
  return 0;
}
