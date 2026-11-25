#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *);
static uint32_t input[2][3][8], output[3][2][8];
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 8; ++k)
        input[i][j][k] = 0x7f800001u + (i * 3 + j) * 8 + k;
  struct tensor a = {input, input, 0, {2, 3, 8}, {24, 8, 1}};
  struct tensor out = {output, output, 0, {3, 2, 8}, {16, 8, 1}};
  _mlir_ciface_compute(&a, &out);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 8; ++k)
        if (input[i][j][k] != output[j][i][k])
          return 1;
  puts("PASS: prefix transpose preserves every FP32 payload bit");
  return 0;
}
