#include <engine.h>
#include <stdint.h>
#include <stdio.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *,
                                 struct matrix *);
static float a[3][768], output[3][19];
static int8_t weights[20][800];
int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  struct matrix am = {a, a, 0, {3, 768}, {768, 1}};
  struct matrix wm = {weights, weights, 0, {20, 800}, {800, 1}};
  struct matrix cm = {output, output, 0, {3, 19}, {19, 1}};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 768; ++k)
      a[i][k] = (i + k) % 5 - 2;
  for (int j = 0; j < 19; ++j) {
    for (int k = 0; k < 768; ++k)
      weights[j][k] = (j + k) % 7 - 3;
    for (int block = 0; block < 24; ++block)
      weights[j][768 + block] = (int8_t)(126 + block % 3);
  }
  _mlir_ciface_compute(&am, &wm, &cm);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 19; ++j) {
      float expected = 0;
      for (int k = 0; k < 768; ++k) {
        int exponent = (k / 32) % 3;
        float scale = exponent == 0 ? 0.5f : exponent == 1 ? 1.0f : 2.0f;
        expected += a[i][k] * weights[j][k] * scale;
      }
      if (output[i][j] != expected) {
        printf("Packed weight mismatch at %d,%d: doubled values %d != %d\n", i,
               j, (int)(output[i][j] * 2), (int)(expected * 2));
        return 1;
      }
    }
  puts("Compiled packed weights: block exponents and output tail passed");
  return 0;
}
