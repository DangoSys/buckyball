#include <engine.h>
#include <stdint.h>
#include <stdio.h>

struct tensor {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *,
                                 struct tensor *);
static float a[2][3][64] __attribute__((aligned(64)));
static float w[2][64][5] __attribute__((aligned(64)));
static float c[2][3][5] __attribute__((aligned(64)));

int main(void) {
  struct engine engine;
  struct engine_slot slots[1];
  engine_init(&engine, slots, 1);
  struct tensor am = {&a[0][0][0], &a[0][0][0], 0, {2, 3, 64}, {192, 64, 1}};
  struct tensor wm = {&w[0][0][0], &w[0][0][0], 0, {2, 64, 5}, {320, 5, 1}};
  struct tensor cm = {&c[0][0][0], &c[0][0][0], 0, {2, 3, 5}, {15, 5, 1}};
  for (int batch = 0; batch < 2; ++batch) {
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 64; ++k)
        a[batch][i][k] = (float)((k + i + batch) % 7 - 3);
    for (int k = 0; k < 64; ++k)
      for (int j = 0; j < 5; ++j)
        w[batch][k][j] = (float)((k + j * 3 + batch) % 5 - 2);
  }
  _mlir_ciface_compute(&am, &wm, &cm);
  for (int batch = 0; batch < 2; ++batch)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 5; ++j) {
        float expected = 0;
        for (int k = 0; k < 64; ++k)
          expected += a[batch][i][k] * w[batch][k][j];
        if (c[batch][i][j] != expected) {
          printf("Attention mismatch: batch %d row %d column %d: %d != %d\n",
                 batch, i, j, (int)c[batch][i][j], (int)expected);
          return 1;
        }
      }
  puts("Compiled attention stream: transposed quantization and five-column "
       "tail passed");
  return 0;
}
