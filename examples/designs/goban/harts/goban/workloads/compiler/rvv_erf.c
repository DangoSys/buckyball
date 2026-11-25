#include <kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *);
static float input[2][36], output[2][36];
static const uint32_t cases[36] = {
    0,          1,          0x007fffff, 0x00800000, 0x03ffffff, 0x04000000,
    0x317fffff, 0x31800000, 0x3d000001, 0x3e000001, 0x3e800001, 0x3f000001,
    0x3f57ffff, 0x3f580000, 0x3f580001, 0x3f7fffff, 0x3f800000, 0x3f9fffff,
    0x3fa00000, 0x3fa00001, 0x3fc00000, 0x40000000, 0x4036db6d, 0x4036db6e,
    0x4036db6f, 0x40400000, 0x40800001, 0x40bfffff, 0x40c00000, 0x40c00001,
    0x7f7fffff, 0x7f800000, 0x7fc00000, 0x7f800001, 0x3eaaaaab, 0x3f234567};
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 36; ++column) {
      uint32_t bits = cases[column] | ((uint32_t)row << 31);
      memcpy(&input[row][column], &bits, 4);
    }
  struct matrix a = {input, input, 0, {2, 36}, {36, 1}};
  struct matrix out = {output, output, 0, {2, 36}, {36, 1}};
  _mlir_ciface_compute(&a, &out);
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 36; ++column) {
      float expected = erff(input[row][column]);
      if (memcmp(&expected, &output[row][column], 4)) {
        uint32_t want, got;
        memcpy(&want, &expected, 4);
        memcpy(&got, &output[row][column], 4);
        printf("RVV erf mismatch at %d,%d: %08x != %08x\n", row, column, got,
               want);
        return 1;
      }
    }
  puts("Compiled RVV erf: reference bits, boundaries and tails passed");
  return 0;
}
