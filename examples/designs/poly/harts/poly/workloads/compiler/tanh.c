#include <kernel.h>
#include <stdint.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
void _mlir_ciface_tanh_rows(struct matrix *, struct matrix *);
static const float samples[33] = {-__builtin_inff(),
                                  -20,
                                  -10,
                                  -8,
                                  -5,
                                  -2,
                                  -1,
                                  -.5f,
                                  -.0004f,
                                  -.0001f,
                                  -1e-8f,
                                  -0.0f,
                                  0.0f,
                                  1e-8f,
                                  .0001f,
                                  .0004f,
                                  .5f,
                                  1,
                                  2,
                                  5,
                                  8,
                                  10,
                                  20,
                                  __builtin_inff(),
                                  __builtin_nanf(""),
                                  -1e-45f,
                                  1e-45f,
                                  -.25f,
                                  .25f,
                                  -3,
                                  3,
                                  7.99f,
                                  -7.99f};
// Independently rounded FP32 tanh reference values.
static const uint32_t expected[33] = {
    0xbf800000, 0xbf800000, 0xbf800000, 0xbf7ffffc, 0xbf7ffa0d, 0xbf76ca83,
    0xbf42f7d6, 0xbeec9a9f, 0xb9d1b716, 0xb8d1b717, 0xb22bcc77, 0x80000000,
    0x00000000, 0x322bcc77, 0x38d1b717, 0x39d1b716, 0x3eec9a9f, 0x3f42f7d6,
    0x3f76ca83, 0x3f7ffa0d, 0x3f7ffffc, 0x3f800000, 0x3f800000, 0x3f800000,
    0x7fc00000, 0x80000001, 0x00000001, 0xbe7acbf5, 0x3e7acbf5, 0xbf7ebbe9,
    0x3f7ebbe9, 0x3f7ffffc, 0xbf7ffffc};
static float input[5 * 37] __attribute__((aligned(64)));
static float output[5 * 39] __attribute__((aligned(64)));

int main(void) {
  struct kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  for (unsigned row = 0; row < 5; ++row) {
    for (unsigned col = 0; col < 33; ++col)
      input[row * 37 + 2 + col] = samples[col];
    for (unsigned col = 0; col < 39; ++col)
      output[row * 39 + col] = 17;
  }
  struct matrix a = {input, input, 2, {5, 33}, {37, 1}};
  struct matrix out = {output, output, 3, {5, 33}, {39, 1}};
  _mlir_ciface_tanh_rows(&a, &out);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 39; ++col) {
      float value = output[row * 39 + col];
      if (col < 3 || col >= 36) {
        if (value != 17)
          return 1;
        continue;
      }
      union {
        uint32_t bits;
        float value;
      } reference = {.bits = expected[col - 3]};
      if (__builtin_isnan(reference.value)) {
        if (!__builtin_isnan(value))
          return 2;
      } else if (!__builtin_isfinite(value) ||
                 __builtin_fabsf(value - reference.value) > 3e-7f)
        return 3;
      if (reference.value == 0 &&
          __builtin_signbit(value) != __builtin_signbit(reference.value))
        return 4;
    }
  return 0;
}
