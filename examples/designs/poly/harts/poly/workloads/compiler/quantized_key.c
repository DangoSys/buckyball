#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *);
extern void _mlir_ciface_dynamic_inner(struct tensor *, struct tensor *);
extern void _mlir_ciface_dynamic_last(struct tensor *, struct tensor *);
extern void _mlir_ciface_static_last(struct tensor *, struct tensor *);
extern void _mlir_ciface_static_wide(struct tensor *, struct tensor *);
extern void _mlir_ciface_static_narrow(struct tensor *, struct tensor *);
extern void _mlir_ciface_static_full(struct tensor *, struct tensor *);
static float input[2 * 65 * 129] __attribute__((aligned(64)));
static int8_t output[2 * 99 * 129] __attribute__((aligned(64)));
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  const int lengths[] = {19, 32, 64, 65};
  for (int run = 0; run < 13; ++run) {
    int rows = run == 12 ? 1 : run >= 10 ? 3 : 129;
    int last = run >= 5, k = run == 12         ? 4096
                             : run == 11       ? 32
                             : run == 10       ? 512
                             : run && run != 9 ? lengths[(run - 1) % 4]
                                               : 64;
    int padded = (k + 31) / 32 * 32, packed = padded + padded / 32;
    struct tensor a = {input,
                       input,
                       0,
                       {2, last ? rows : k, last ? k : rows},
                       {rows * k, last ? k : rows, 1}};
    struct tensor q = {output,
                       output,
                       0,
                       {2, last ? rows : packed, last ? packed : rows},
                       {rows * packed, last ? packed : rows, 1}};
    for (int i = 0; i < 2 * rows * k; ++i)
      input[i] = (i % 2 ? -1 : 1) * (2 + i % 2) *
                 (run >= 10 ? 1 << ((i % k / 32) % 4) : 1);
    if (run == 12)
      _mlir_ciface_static_full(&a, &q);
    else if (run == 11)
      _mlir_ciface_static_narrow(&a, &q);
    else if (run == 10)
      _mlir_ciface_static_wide(&a, &q);
    else if (!run)
      _mlir_ciface_compute(&a, &q);
    else if (run == 9)
      _mlir_ciface_static_last(&a, &q);
    else if (last)
      _mlir_ciface_dynamic_last(&a, &q);
    else
      _mlir_ciface_dynamic_inner(&a, &q);
    for (int batch = 0; batch < 2; ++batch)
      for (int t = 0; t < packed; ++t)
        for (int j = 0; j < rows; ++j) {
          int source = batch * rows * k + (last ? j * k + t : t * rows + j);
          int target =
              batch * rows * packed + (last ? j * packed + t : t * rows + j);
          int scale = run >= 10 ? 1 << ((t / 32) % 4) : 1;
          int expected = t < k ? (int)input[source] * 32 / scale
                         : t < padded
                             ? 0
                             : 122 + (run >= 10 ? (t - padded) % 4 : 0);
          if (output[target] != expected) {
            printf("Quantization mismatch: run %d at %d,%d,%d: %d != %d\n", run,
                   batch, t, j, output[target], expected);
            return 1;
          }
        }
  }
  puts("Compiled quantization: static and dynamic axes, codes, padding and "
       "exponents passed");
  return 0;
}
