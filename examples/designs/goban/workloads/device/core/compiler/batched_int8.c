#include <engine.h>
#include <stdint.h>
#include <stdio.h>

struct tensor {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *,
                                 struct tensor *);
static float a[2][3][32] __attribute__((aligned(64)));
static float w[2][32][16] __attribute__((aligned(64)));
static float c[2][3][16] __attribute__((aligned(64)));

int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  struct tensor am = {&a[0][0][0], &a[0][0][0], 0, {2, 3, 32}, {96, 32, 1}};
  struct tensor wm = {&w[0][0][0], &w[0][0][0], 0, {2, 32, 16}, {512, 16, 1}};
  struct tensor cm = {&c[0][0][0], &c[0][0][0], 0, {2, 3, 16}, {48, 16, 1}};
  for (int batch = 0; batch < 2; ++batch) {
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 32; ++k)
        a[batch][i][k] = (float)((k + i + batch) % 7 - 3);
    for (int k = 0; k < 32; ++k)
      for (int j = 0; j < 16; ++j)
        w[batch][k][j] = (float)((k + j * 3 + batch) % 5 - 2);
  }
  _mlir_ciface_compute(&am, &wm, &cm);
  for (int batch = 0; batch < 2; ++batch)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 16; ++j) {
        float expected = 0;
        for (int k = 0; k < 32; ++k)
          expected += a[batch][i][k] * w[batch][k][j];
        if (c[batch][i][j] != expected) {
          printf("Attention mismatch: batch %d row %d column %d: %d != %d\n",
                 batch, i, j, (int)c[batch][i][j], (int)expected);
          return 1;
        }
      }
  puts("Compiled INT8 attention: online quantization of both operands passed");
  return 0;
}
