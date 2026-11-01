#include <engine.h>
#include <stdint.h>
#include <stdio.h>
struct vector {
  void *allocated, *aligned;
  int64_t offset, size, stride;
};
extern void _mlir_ciface_compute(struct vector *, struct vector *,
                                 struct vector *);
static int32_t input[19];
static float scale[19], output[19];
int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  struct vector x = {input, input, 0, 19, 1};
  struct vector s = {scale, scale, 0, 19, 1};
  struct vector y = {output, output, 0, 19, 1};
  for (int i = 0; i < 19; ++i) {
    input[i] = i % 2 ? -(16777217 + i) : i * 31;
    scale[i] = i % 3 ? 0.25f : 0.5f;
  }
  _mlir_ciface_compute(&x, &s, &y);
  for (int i = 0; i < 19; ++i)
    if (output[i] != (float)input[i] * scale[i]) {
      printf("Bias mismatch at %d\n", i);
      return 1;
    }
  puts("Compiled bias: rounding, channel scales and nineteen-element tail "
       "passed");
  return 0;
}
