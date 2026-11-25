#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *);
static float input[2][32], output[2][16];
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 32; ++j)
      input[i][j] = i * 32 + j - 15;
  struct tensor a = {input, input, 0, {1, 2, 32}, {64, 32, 1}};
  struct tensor out = {output, output, 0, {1, 2, 16}, {32, 16, 1}};
  _mlir_ciface_compute(&a, &out);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 16; ++j)
      if (output[i][j] != input[i][j] * input[i][j + 16])
        return 1;
  puts("PASS: two GP inputs share a source with distinct row offsets");
  return 0;
}
