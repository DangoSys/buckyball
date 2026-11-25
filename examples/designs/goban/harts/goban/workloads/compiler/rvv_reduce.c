#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct tensor *, struct matrix *);
static float input[16][36], output[16], expected[16];
static const float pattern[8] = {16777216.0f, 1.0f,    -16777216.0f, 1.0f,
                                 0.25f,       -0.125f, 0.5f,         -0.25f};

int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (int row = 0; row < 16; ++row) {
    float sum = 0;
    for (int column = 0; column < 36; ++column) {
      float value = pattern[(column + row) % 8];
      if (row & 1)
        value = -value;
      input[row][column] = value;
      sum = value + sum;
    }
    expected[row] = sum;
  }
  struct tensor a = {input, input, 0, {16, 1, 36}, {36, 36, 1}};
  struct matrix out = {output, output, 0, {16, 1}, {1, 1}};
  _mlir_ciface_compute(&a, &out);
  if (memcmp(output, expected, sizeof(output))) {
    puts("Compiled RVV reduction: ordered FP32 sums differ");
    return 1;
  }
  puts("Compiled RVV reduction: cancellation, chunk boundary and row groups "
       "passed");
  return 0;
}
