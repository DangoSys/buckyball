#include <CRunnerUtils.h>
#include <cstdint>
extern "C" void _mlir_ciface_compute(StridedMemRefType<int8_t, 1> *,
                                     StridedMemRefType<int8_t, 1> *,
                                     StridedMemRefType<int8_t, 1> *);
static constexpr auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t a[48], b[2][32 * 48];
    alignas(64) uint8_t expected[336];
  } v{};
  v.a[0] = 1;
  v.a[32] = 127;
  for (int t = 0; t < 2; ++t)
    for (int j = 0; j < 32; ++j) {
      v.b[t][j * 48] = j == 31  ? (t ? -1 : 1)
                       : t      ? 0
                       : j == 0 ? 1
                       : j == 1 ? 3
                       : j == 2 ? -1
                       : j == 3 ? -3
                                : 0;
      v.b[t][j * 48 + 32] = j == 31 ? 160 : 127;
    }
  v.expected[0] = 1;
  v.expected[2] = 33;
  v.expected[4] = 1;
  for (int j = 0; j < 32; ++j) {
    uint64_t x = uint64_t(j == 1    ? 2
                          : j == 3  ? -2
                          : j == 31 ? INT64_C(1) << 32
                                    : 0);
    for (int byte = 0; byte < 8; ++byte)
      v.expected[16 + j * 8 + byte] = x >> (byte * 8);
  }
  v.expected[273] = 64;
  v.expected[275] = uint8_t(-64);
  v.expected[304] = 123;
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
       *bp = const_cast<int8_t *>(fixture.b[0]);
  auto *op = reinterpret_cast<int8_t *>(output.bytes);
  StridedMemRefType<int8_t, 1> a{ap, ap, 0, {48}, {1}};
  StridedMemRefType<int8_t, 1> b{bp, bp, 0, {3072}, {1}};
  StridedMemRefType<int8_t, 1> out{op, op, 0, {336}, {1}};
  _mlir_ciface_compute(&a, &b, &out);
  return 0;
}
