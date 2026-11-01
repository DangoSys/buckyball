#include <engine.h>
#include <stdint.h>
#include <stdio.h>

struct tensor {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *,
                                 struct tensor *);
static float a[17][2048] __attribute__((aligned(64)));
static float w[2048][17] __attribute__((aligned(64)));
static float c[17][17] __attribute__((aligned(64)));

int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  struct tensor am = {a[0], a[0], 0, {17, 2048}, {2048, 1}};
  struct tensor wm = {w[0], w[0], 0, {2048, 17}, {17, 1}};
  struct tensor cm = {c[0], c[0], 0, {17, 17}, {17, 1}};
  for (int i = 0; i < 17; ++i)
    for (int k = 0; k < 2048; ++k) {
      a[i][k] = (float)((k + i) % 7 - 3);
      w[k][i] = (float)((k + i * 3) % 5 - 2);
    }
  _mlir_ciface_compute(&am, &wm, &cm);
  for (int i = 0; i < 17; ++i)
    for (int j = 0; j < 17; ++j) {
      float expected = 0;
      for (int k = 0; k < 2048; ++k)
        expected += a[i][k] * w[k][j];
      if (c[i][j] != expected) {
        printf("K block mismatch at %d,%d: %d != %d\n", i, j, (int)c[i][j],
               (int)expected);
        return 1;
      }
    }
  return 0;
}
