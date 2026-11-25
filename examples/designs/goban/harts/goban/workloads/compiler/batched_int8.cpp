#include <kernel.h>
#include <stdint.h>

struct tensor {
  const float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" void _mlir_ciface_compute(struct tensor *, struct tensor *,
                                     struct tensor *);
struct Fixture {
  float a[2][65][32], w[2][32][4], expected[2][65][4];
};
alignas(64) static constexpr Fixture fixture = [] {
  Fixture f{};
  for (int batch = 0; batch < 2; ++batch) {
    for (int i = 0; i < 65; ++i)
      for (int k = 0; k < 32; ++k)
        f.a[batch][i][k] = float((k + i + batch) % 7 - 3);
    for (int k = 0; k < 32; ++k)
      for (int j = 0; j < 4; ++j)
        f.w[batch][k][j] = float((k + j * 3 + batch) % 5 - 2);
    for (int i = 0; i < 65; ++i)
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 32; ++k)
          f.expected[batch][i][j] += f.a[batch][i][k] * f.w[batch][k][j];
  }
  return f;
}();
alignas(64) static float c[2][65][4];
__attribute__((used, section(".bbsim.check_output"),
               aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {c, fixture.expected, sizeof(c)};

int main() {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  struct tensor am = {
      &fixture.a[0][0][0], &fixture.a[0][0][0], 0, {2, 65, 32}, {2080, 32, 1}};
  struct tensor wm = {
      &fixture.w[0][0][0], &fixture.w[0][0][0], 0, {2, 32, 4}, {128, 4, 1}};
  struct tensor cm = {&c[0][0][0], &c[0][0][0], 0, {2, 65, 4}, {260, 4, 1}};
  _mlir_ciface_compute(&am, &wm, &cm);
  auto *exit = reinterpret_cast<volatile uint32_t *>(uintptr_t(0x60000000));
  *exit = UINT32_C(0x42424f55);
  for (unsigned offset = 0; offset < sizeof(c); offset += 4) {
    uint32_t word;
    __builtin_memcpy(&word, reinterpret_cast<const char *>(c) + offset, 4);
    *exit = word;
  }
  return 0;
}
