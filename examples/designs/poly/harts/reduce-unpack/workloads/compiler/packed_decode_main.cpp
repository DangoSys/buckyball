#include <cstdint>
struct Matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern "C" void _mlir_ciface_compute32(Matrix *, Matrix *);
extern "C" void _mlir_ciface_compute544(Matrix *, Matrix *);
extern "C" void _mlir_ciface_compute4096(Matrix *, Matrix *);
template <unsigned W> struct Case {
  static constexpr unsigned P = W + W / 32, S = W + 8;
  alignas(64) static constexpr auto fixture = [] {
    struct Fixture {
      uint8_t input[3 * P + 14];
      uint32_t expected[2][3 * S];
    } f{};
    for (auto &v : f.input)
      v = 0xaf;
    for (auto &run : f.expected)
      for (auto &v : run)
        v = 0xa5a5a5a5;
    for (unsigned row = 0; row < 3; ++row) {
      for (unsigned block = 0; block < W / 32; ++block) {
        unsigned exponent = (block * 37 + row * 127) % 256;
        f.input[7 + row * P + W + block] = exponent;
        for (unsigned lane = 0; lane < 32; ++lane) {
          int code = int((lane * 17 + row * 13 + block) % 256) - 128;
          f.input[7 + row * P + 32 * block + lane] = uint8_t(code);
          unsigned a = code < 0 ? -code : code, l = 0;
          for (unsigned v = a; v > 1; v >>= 1)
            ++l;
          unsigned field = exponent + l;
          uint32_t sign = code < 0 ? 0x80000000 : 0;
          uint32_t value = a == 0         ? 0
                           : field >= 255 ? sign | 0x7f800000
                           : field == 0   ? sign | 0x00400000
                                          : sign | (field << 23) |
                                              ((a << (23 - l)) & 0x7fffff);
          f.expected[1][4 + row * S + 32 * block + lane] = value;
          if (row == 0)
            f.expected[0][4 + 32 * block + lane] = value;
        }
      }
    }
    return f;
  }();
  alignas(64) static inline auto output = [] {
    struct Output {
      uint32_t words[2][3 * S];
    } f{};
    for (auto &run : f.words)
      for (auto &v : run)
        v = 0xa5a5a5a5;
    return f;
  }();
};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} checks[] __attribute__((section(".bbsim.check"), used, retain)) = {
    {Case<32>::output.words, Case<32>::fixture.expected,
     sizeof(Case<32>::output)},
    {Case<544>::output.words, Case<544>::fixture.expected,
     sizeof(Case<544>::output)},
    {Case<4096>::output.words, Case<4096>::fixture.expected,
     sizeof(Case<4096>::output)}};
template <unsigned W> void run(void (*compute)(Matrix *, Matrix *)) {
  auto *data = const_cast<uint8_t *>(Case<W>::fixture.input);
  for (int trial = 0; trial < 2; ++trial) {
    int64_t rows = trial == 0 ? 1 : 3;
    Matrix input{data, data, 7, {rows, Case<W>::P}, {Case<W>::P, 1}};
    Matrix output{Case<W>::output.words[trial],
                  Case<W>::output.words[trial],
                  4,
                  {rows, W},
                  {Case<W>::S, 1}};
    compute(&input, &output);
  }
}
int main() {
  run<32>(_mlir_ciface_compute32);
  run<544>(_mlir_ciface_compute544);
  run<4096>(_mlir_ciface_compute4096);
  return 0;
}
