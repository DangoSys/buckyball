#include <bit>
#include <kernel.h>
#include <stdint.h>

struct matrix {
  const float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct vector {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern "C" void _mlir_ciface_compute(matrix *, vector *);
struct Fixture {
  float values[8][36];
};
alignas(64) static constexpr Fixture input = [] {
  Fixture f{};
  for (int column = 0; column < 36; ++column) {
    f.values[0][column] = float(column % 7 - 3);
    f.values[1][column] = -0.0f;
    f.values[2][column] = -0.0f;
    f.values[3][column] = std::bit_cast<float>(UINT32_C(0xff800000));
    for (int row = 4; row < 8; ++row)
      f.values[row][column] = 1.0f;
  }
  f.values[2][1] = 0.0f;
  f.values[4][0] = std::bit_cast<float>(UINT32_C(0x7fc00000));
  f.values[5][1] = std::bit_cast<float>(UINT32_C(0x7f800000));
  f.values[6][35] = 100.0f;
  f.values[7][35] = std::bit_cast<float>(UINT32_C(0x7fc00000));
  return f;
}();
static const uint32_t expected[] = {0x40400000, 0x80000000, 0,
                                    0xff7fffff, 0x7fc00000, 0x7f800000,
                                    0x42c80000, 0x7fc00000};
alignas(64) static float output[8];
__attribute__((used, section(".bbsim.check_output"),
               aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {output, expected, sizeof(output)};

int main() {
  kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  matrix a{&input.values[0][0], &input.values[0][0], 0, {8, 36}, {36, 1}};
  vector out{output, output, 0, {8}, {1}};
  _mlir_ciface_compute(&a, &out);
  auto *exit = reinterpret_cast<volatile uint32_t *>(uintptr_t(0x60000000));
  *exit = UINT32_C(0x42424f55);
  for (unsigned i = 0; i < 8; ++i) {
    uint32_t word;
    __builtin_memcpy(&word, output + i, 4);
    *exit = word;
  }
  return 0;
}
