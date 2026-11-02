#include <engine.h>
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
static float input[2 * 65 * 129] __attribute__((aligned(64)));
static int8_t output[2 * 99 * 129] __attribute__((aligned(64)));
int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  const int lengths[] = {19, 32, 64, 65};
  for (int run = 0; run < 10; ++run) {
    int last = run >= 5, k = run && run != 9 ? lengths[(run - 1) % 4] : 64;
    int padded = (k + 31) / 32 * 32, packed = padded + padded / 32;
    struct tensor a = {input,
                       input,
                       0,
                       {2, last ? 129 : k, last ? k : 129},
                       {129 * k, last ? k : 129, 1}};
    struct tensor q = {output,
                       output,
                       0,
                       {2, last ? 129 : packed, last ? packed : 129},
                       {129 * packed, last ? packed : 129, 1}};
    for (int i = 0; i < 2 * 129 * k; ++i)
      input[i] = (i % 2 ? -1 : 1) * (2 + i % 2);
    if (!run)
      _mlir_ciface_compute(&a, &q);
    else if (run == 9)
      _mlir_ciface_static_last(&a, &q);
    else if (last)
      _mlir_ciface_dynamic_last(&a, &q);
    else
      _mlir_ciface_dynamic_inner(&a, &q);
    for (int batch = 0; batch < 2; ++batch)
      for (int t = 0; t < packed; ++t)
        for (int j = 0; j < 129; ++j) {
          int source = batch * 129 * k + (last ? j * k + t : t * 129 + j);
          int target =
              batch * 129 * packed + (last ? j * packed + t : t * 129 + j);
          int expected = t < k ? (int)input[source] * 32 : t < padded ? 0 : 122;
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
