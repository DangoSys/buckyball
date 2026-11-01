#include <engine.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *);
static float input[2][64][5] __attribute__((aligned(64)));
static int8_t output[2][66][5] __attribute__((aligned(64)));
int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  struct tensor a = {input, input, 0, {2, 64, 5}, {320, 5, 1}};
  struct tensor q = {output, output, 0, {2, 66, 5}, {330, 5, 1}};
  for (int batch = 0; batch < 2; ++batch)
    for (int k = 0; k < 64; ++k)
      for (int j = 0; j < 5; ++j)
        input[batch][k][j] = (float)((batch + k + j) % 7 - 3);
  _mlir_ciface_compute(&a, &q);
  for (int batch = 0; batch < 2; ++batch)
    for (int k = 0; k < 66; ++k)
      for (int j = 0; j < 5; ++j) {
        int expected = k < 64 ? (int)input[batch][k][j] * 32 : 122;
        if (output[batch][k][j] != expected) {
          printf("Quantized key mismatch at %d,%d,%d: %d != %d\n", batch, k, j,
                 output[batch][k][j], expected);
          return 1;
        }
      }
  puts("Compiled key quantization: codes and exponents on the inner axis "
       "passed");
  return 0;
}
