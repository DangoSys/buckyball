#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct vector {
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern "C" void _mlir_ciface_compute(matrix *, vector *);
extern "C" void _mlir_ciface_maximum(matrix *, vector *);
struct Fixture {
  uint32_t input[3 * 65];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  for (unsigned i = 0; i < 65; ++i) {
    f.input[i] = i % 2 ? 0x3f000000 : 0xbf800000;
    f.input[65 + i] = i % 2 ? 0 : 0x80000000;
    f.input[130 + i] = 0x3f800000;
  }
  f.input[0] = 0x60ad78ec;
  f.input[1] = 0x3f800000;
  f.input[2] = 0xe0ad78ec;
  f.input[130] = 0x7fc12345;
  return f;
}();
alignas(64) static uint32_t output[3];

int main() {
  auto *data = const_cast<uint32_t *>(fixture.input);
  for (int64_t width : {1, 3, 7, 32, 36, 65}) {
    matrix input{data, data, 0, {3, width}, {65, 1}};
    vector result{output, output, 0, {3}, {1}};
    _mlir_ciface_compute(&input, &result);
    for (unsigned row = 0; row < 3; ++row) {
      float sum = 0;
      for (unsigned k = 0; k < width; ++k) {
        float x;
        memcpy(&x, &fixture.input[row * 65 + k], 4);
        asm volatile("fadd.s %0, %1, %2, rne" : "=f"(sum) : "f"(sum), "f"(x));
      }
      uint32_t bits;
      memcpy(&bits, &sum, 4);
      if (output[row] != bits) {
        printf("GP ordered reduction mismatch at %u: %08x != %08x\n", row,
               output[row], bits);
        return 1;
      }
    }
    _mlir_ciface_maximum(&input, &result);
    for (unsigned row = 0; row < 3; ++row) {
      uint32_t bits = 0xff800000;
      float maximum;
      memcpy(&maximum, &bits, 4);
      bool nan = false;
      for (unsigned k = 0; k < width; ++k) {
        uint32_t raw = fixture.input[row * 65 + k];
        nan |= (raw & 0x7fffffff) > 0x7f800000;
        float x;
        memcpy(&x, &raw, 4);
        asm volatile("fmax.s %0, %1, %2"
                     : "=f"(maximum)
                     : "f"(maximum), "f"(x));
      }
      memcpy(&bits, &maximum, 4);
      if (nan)
        bits = 0x7fc00000;
      if (output[row] != bits) {
        printf("GP maximum mismatch at width %ld row %u: %08x != %08x\n", width,
               row, output[row], bits);
        return 1;
      }
    }
  }
  return 0;
}
