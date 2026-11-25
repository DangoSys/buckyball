#include <cstdint>
constexpr unsigned B = 1, K = 32 * B, INPUT_BYTES = (33 * B + 15) / 16 * 16;
static constexpr auto fixture = [] {
  struct {
    alignas(64) uint8_t input[INPUT_BYTES];
    uint32_t expected[K + 8];
  } f{};
  for (auto &x : f.input)
    x = 0xd3;
  for (auto &x : f.expected)
    x = 0xa5a5a5a5;
  for (unsigned block = 0; block < B; ++block) {
    unsigned exponent = 0;
    f.input[K + block] = exponent;
    for (unsigned lane = 0; lane < 32; ++lane) {
      const int edge[] = {0, -1, 1, -128, 127};
      int code =
          lane < 5 ? edge[lane] : int((lane * 17 + block * 7) % 256) - 128;
      f.input[32 * block + lane] = uint8_t(code);
      unsigned a = code < 0 ? -code : code, l = 0;
      for (unsigned v = a; v > 1; v >>= 1)
        ++l;
      uint32_t sign = code < 0 ? 0x80000000 : 0;
      unsigned field = exponent + l;
      f.expected[4 + 32 * block + lane] =
          a == 0         ? 0
          : field >= 255 ? sign | 0x7f800000
          : field == 0   ? sign | 0x00400000
                         : sign | (field << 23) | ((a << (23 - l)) & 0x7fffff);
    }
  }
  return f;
}();
alignas(64) static auto output = [] {
  struct {
    uint32_t words[K + 8];
  } x{};
  for (auto &w : x.words)
    w = 0xa5a5a5a5;
  return x;
}();
alignas(64) static uint8_t source_copy[INPUT_BYTES];
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} checks[] __attribute__((section(".bbsim.check"), used, retain)) = {
    {output.words, fixture.expected, sizeof(output)},
    {source_copy, fixture.input, sizeof(source_copy)}};
template <typename T> struct Memref {
  T *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern "C" void _mlir_ciface_compute(Memref<uint8_t> *, Memref<uint32_t> *,
                                     Memref<uint8_t> *);
int main() {
  auto *data = const_cast<uint8_t *>(fixture.input);
  Memref<uint8_t> input{data, data, 0, {INPUT_BYTES}, {1}};
  Memref<uint32_t> out{output.words + 4, output.words + 4, 0, {K}, {1}};
  Memref<uint8_t> copy{source_copy, source_copy, 0, {INPUT_BYTES}, {1}};
  _mlir_ciface_compute(&input, &out, &copy);
  return 0;
}
