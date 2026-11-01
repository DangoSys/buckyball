#include <engine.h>
#include <stdint.h>
#include <stdio.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct blocks {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct matrix *, struct blocks *,
                                 struct matrix *);
static float a[3][2048], output[3][19];
static int8_t weights[2][20][1056];
int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  struct matrix am = {a, a, 0, {3, 2048}, {2048, 1}};
  struct blocks wm = {weights, weights, 0, {2, 20, 1056}, {21120, 1056, 1}};
  struct matrix cm = {output, output, 0, {3, 19}, {19, 1}};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 2048; ++k)
      a[i][k] = (i + k) % 5 - 2;
  for (int block = 0; block < 2; ++block)
    for (int j = 0; j < 19; ++j) {
      for (int k = 0; k < 1024; ++k)
        weights[block][j][k] = (j + block * 1024 + k) % 7 - 3;
      for (int k = 0; k < 32; ++k)
        weights[block][j][1024 + k] = 126 + (block * 32 + k) % 3;
    }
  _mlir_ciface_compute(&am, &wm, &cm);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 19; ++j) {
      float expected = 0;
      for (int k = 0; k < 2048; ++k) {
        int exponent = (k / 32) % 3;
        float scale = exponent == 0 ? 0.5f : exponent == 1 ? 1.0f : 2.0f;
        expected += a[i][k] * ((j + k) % 7 - 3) * scale;
      }
      if (output[i][j] != expected) {
        printf("Offline K block mismatch at %d,%d\n", i, j);
        return 1;
      }
    }
  return 0;
}
