#include <engine.h>
#include <stdint.h>
#include <stdio.h>

struct matrix {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *,
                                 struct matrix *);
static float a[5][768] __attribute__((aligned(64)));
static float w[32][768] __attribute__((aligned(64)));
static float c[5][32] __attribute__((aligned(64)));

int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  struct matrix am = {&a[0][0], &a[0][0], 0, {3, 768}, {768, 1}};
  struct matrix wm = {&w[0][0], &w[0][0], 0, {32, 768}, {768, 1}};
  struct matrix cm = {&c[0][0], &c[0][0], 0, {3, 32}, {32, 1}};
  for (int run = 0; run < 2; ++run) {
    int rows = run ? 5 : 3;
    am.sizes[0] = cm.sizes[0] = rows;
    for (int i = 0; i < rows; ++i)
      for (int k = 0; k < 768; ++k)
        a[i][k] = (float)((k + i + run) % 7 - 3);
    for (int j = 0; j < 32; ++j)
      for (int k = 0; k < 768; ++k)
        w[j][k] = (float)((k + j * 3) % 5 - 2);
    _mlir_ciface_compute(&am, &wm, &cm);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < 32; ++j) {
        float expected = 0;
        for (int k = 0; k < 768; ++k)
          expected += a[i][k] * w[j][k];
        if (c[i][j] != expected) {
          printf("MXINT8 mismatch: run %d row %d column %d: %d != %d\n", run, i,
                 j, (int)c[i][j], (int)expected);
          return 1;
        }
      }
  }
  puts("MXINT8 compiled stream: quantization, tiled accumulation and replay "
       "passed");
  return 0;
}
