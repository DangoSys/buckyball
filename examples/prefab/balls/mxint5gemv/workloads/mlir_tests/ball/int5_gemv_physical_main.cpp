#include <CRunnerUtils.h>
#include <cstdint>
extern "C" void _mlir_ciface_compute(StridedMemRefType<int8_t, 1> *,
                                     StridedMemRefType<int8_t, 1> *,
                                     StridedMemRefType<int8_t, 1> *);

static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[528];
    alignas(64) uint8_t b[32 * 352];
    alignas(64) uint8_t expected[336];
  } v{};
  for (unsigned k = 0; k < 528; ++k)
    v.a[k] = k < 512 ? 1 : 127;
  for (unsigned j = 0; j < 32; ++j) {
    const int code = int(j % 7) - 3;
    const int scale = int(j % 3) - 1;
    for (unsigned k = 0; k < 512; ++k) {
      const unsigned bit = 5 * k;
      const unsigned encoded = unsigned(code) & 31;
      v.b[j * 352 + bit / 8] |= uint8_t(encoded << (bit % 8));
      if (bit % 8 > 3)
        v.b[j * 352 + bit / 8 + 1] |= uint8_t(encoded >> (8 - bit % 8));
    }
    for (unsigned block = 0; block < 16; ++block) {
      v.b[j * 352 + 320 + block] = 17;
      v.b[j * 352 + 336 + block] = uint8_t(127 + scale);
    }
    const uint64_t value =
        uint64_t(int64_t(512 * code * 17) * (INT64_C(1) << (27 + scale)));
    for (unsigned byte = 0; byte < 8; ++byte)
      v.expected[16 + j * 8 + byte] = uint8_t(value >> (8 * byte));
    const int scaled = 17 * code;
    const unsigned magnitude = unsigned(scaled < 0 ? -scaled : scaled);
    const unsigned half_even =
        magnitude / 2 + ((magnitude & 1) && ((magnitude / 2) & 1));
    v.expected[272 + j] =
        uint8_t(scale < 0 ? (scaled < 0 ? -int(half_even) : int(half_even))
                : scale   ? 2 * scaled
                          : scaled);
  }
  v.expected[0] = 16;
  v.expected[2] = 0xfd;
  v.expected[3] = 0xff;
  v.expected[4] = 1;
  v.expected[5] = 5;
  v.expected[6] = 28;
  v.expected[304] = 133;
  for (unsigned i = 320; i < 336; ++i)
    v.expected[i] = 0x5a;
  return v;
}();

alignas(64) static auto output = []() constexpr {
  struct {
    uint8_t bytes[336];
  } v{};
  for (auto &byte : v.bytes)
    byte = 0x5a;
  return v;
}();

static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check"), used, retain)) = {
    output.bytes, fixture.expected, sizeof(output)};

int main() {
  auto *ap = const_cast<int8_t *>(fixture.a);
  auto *bp = reinterpret_cast<int8_t *>(const_cast<uint8_t *>(fixture.b));
  auto *op = reinterpret_cast<int8_t *>(output.bytes);
  StridedMemRefType<int8_t, 1> a{ap, ap, 0, {528}, {1}};
  StridedMemRefType<int8_t, 1> b{bp, bp, 0, {11264}, {1}};
  StridedMemRefType<int8_t, 1> out{op, op, 0, {336}, {1}};
  _mlir_ciface_compute(&a, &b, &out);
  return 0;
}
