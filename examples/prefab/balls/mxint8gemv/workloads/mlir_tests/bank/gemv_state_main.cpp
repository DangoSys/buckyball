#include <CRunnerUtils.h>
#include <cstdint>
extern "C" void _mlir_ciface_compute(StridedMemRefType<int8_t, 1> *,
                                     StridedMemRefType<int8_t, 1> *,
                                     StridedMemRefType<int8_t, 1> *);
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[528], b[32 * 528];
    alignas(64) uint8_t expected[336];
  } v{};
  for (int k = 0; k < 528; ++k)
    v.a[k] = k < 512 ? 1 : 127;
  for (int j = 0; j < 32; ++j) {
    int code = j % 7 - 3, scale = j % 3 - 1;
    for (int k = 0; k < 528; ++k)
      v.b[j * 528 + k] = k < 512 ? code : 127 + scale;
    uint64_t value =
        uint64_t(int64_t(512 * code) * (INT64_C(1) << (31 + scale)));
    for (int byte = 0; byte < 8; ++byte)
      v.expected[16 + j * 8 + byte] = value >> (8 * byte);
    v.expected[272 + j] = uint8_t(code * (scale < 0 ? 8 : scale ? 32 : 16));
  }
  v.expected[0] = 16;
  v.expected[2] = 1;
  v.expected[4] = 1;
  v.expected[304] = 133;
  for (int i = 320; i < 336; ++i)
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
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output.bytes, fixture.expected, sizeof(output)};
int main() {
  auto *ap = const_cast<int8_t *>(fixture.a),
       *bp = const_cast<int8_t *>(fixture.b);
  auto *op = reinterpret_cast<int8_t *>(output.bytes);
  StridedMemRefType<int8_t, 1> a{ap, ap, 0, {528}, {1}};
  StridedMemRefType<int8_t, 1> b{bp, bp, 0, {16896}, {1}};
  StridedMemRefType<int8_t, 1> out{op, op, 0, {336}, {1}};
  _mlir_ciface_compute(&a, &b, &out);
  return 0;
}
