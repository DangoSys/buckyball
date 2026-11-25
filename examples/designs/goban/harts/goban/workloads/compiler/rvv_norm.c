#include <kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *,
                                 struct matrix *);
static float input[2][36], bias[36], output[2][36];

int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (int column = 0; column < 36; ++column) {
    bias[column] = (column % 11 - 5) * 0.125f;
    for (int row = 0; row < 2; ++row) {
      uint32_t bits = UINT32_C(0x3e800001) + column * 7919 + row * 313;
      if (column & 1)
        bits |= UINT32_C(0x80000000);
      memcpy(&input[row][column], &bits, 4);
    }
  }
  struct matrix a = {input, input, 0, {2, 36}, {36, 1}};
  struct matrix b = {bias, bias, 0, {1, 36}, {36, 1}};
  struct matrix out = {output, output, 0, {2, 36}, {36, 1}};
  _mlir_ciface_compute(&a, &b, &out);
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 36; ++column) {
      volatile float denominator = fabsf(input[row][column]) + 0.125f;
      volatile float inverse = 1.0f / sqrtf(denominator);
      volatile float scaled = input[row][column] * inverse;
      float expected = scaled + bias[column];
      if (memcmp(&expected, &output[row][column], 4)) {
        printf("Compiled RVV normalization mismatch at row %d column %d\n", row,
               column);
        return 1;
      }
    }
  puts("Compiled RVV normalization: absolute, sqrt and reciprocal passed");
  return 0;
}
