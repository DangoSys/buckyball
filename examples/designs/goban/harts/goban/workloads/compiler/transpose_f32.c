#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *);
static uint32_t input[8][44], output[44][8], expected[44][8];

int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (int row = 0; row < 8; ++row)
    for (int column = 0; column < 44; ++column) {
      uint32_t bits = UINT32_C(0x3f000001) + (row * 44 + column) * 7919;
      if (column % 7 == 0)
        bits = UINT32_C(0x7f800001) + row;
      if (column % 11 == 0)
        bits = 0;
      if (column & 1)
        bits |= UINT32_C(0x80000000);
      input[row][column] = expected[column][row] = bits;
    }
  struct matrix a = {input, input, 0, {8, 44}, {44, 1}};
  struct matrix out = {output, output, 0, {44, 8}, {8, 1}};
  _mlir_ciface_compute(&a, &out);
  if (memcmp(output, expected, sizeof(output))) {
    puts("Compiled FP32 transpose: output bits differ");
    return 1;
  }
  puts(
      "Compiled FP32 transpose: strided groups, tail tile and NaN bits passed");
  return 0;
}
