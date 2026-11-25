#include <kernel.h>
#include <stdint.h>
#include <stdio.h>

struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_static_stride(struct tensor *, struct tensor *);
static float input[2 * 32 * 1025] __attribute__((aligned(64)));
static int8_t output[2 * 33 * 1025] __attribute__((aligned(64)));

int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  struct tensor a = {input, input, 0, {2, 32, 1025}, {32 * 1025, 1025, 1}};
  struct tensor q = {output, output, 0, {2, 33, 1025}, {33 * 1025, 1025, 1}};
  for (unsigned i = 0; i < sizeof(input) / sizeof(*input); ++i)
    input[i] = i % 2 ? -3 : 2;
  _mlir_ciface_static_stride(&a, &q);
  for (int b = 0; b < 2; ++b)
    for (int k = 0; k < 33; ++k)
      for (int n = 0; n < 1025; ++n) {
        int expected = k < 32 ? (int)input[(b * 32 + k) * 1025 + n] * 32 : 122;
        int actual = output[(b * 33 + k) * 1025 + n];
        if (actual != expected) {
          printf("Strided quantization mismatch at %d,%d,%d: %d != %d\n", b, k,
                 n, actual, expected);
          return 1;
        }
      }
  puts("Compiled strided quantization passed");
  return 0;
}
