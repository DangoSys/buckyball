#include <kernel.h>
#include <stdint.h>
#include <stdio.h>

struct matrix {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *,
                                 struct matrix *, struct matrix *);
static float a[3][768] __attribute__((aligned(64)));
static float w[36][768] __attribute__((aligned(64)));
static float bias[36] __attribute__((aligned(64)));
static float c[3][36] __attribute__((aligned(64)));

int main(void) {
  struct kernel kernel;
  struct kernel_slot slots[4];
  kernel_init(&kernel, slots, 4);
  struct matrix am = {&a[0][0], &a[0][0], 0, {3, 768}, {768, 1}};
  struct matrix wm = {&w[0][0], &w[0][0], 0, {36, 768}, {768, 1}};
  struct matrix bm = {&bias[0], &bias[0], 0, {1, 36}, {36, 1}};
  struct matrix cm = {&c[0][0], &c[0][0], 0, {3, 36}, {36, 1}};
  for (int run = 0; run < 2; ++run) {
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 768; ++k)
        a[i][k] = (float)((k + i + run) % 7 - 3);
    for (int j = 0; j < 36; ++j) {
      bias[j] = (float)(j * 5 - 70 + run);
      for (int k = 0; k < 768; ++k)
        w[j][k] = (float)((k + j * 3) % 5 - 2);
    }
    _mlir_ciface_compute(&am, &wm, &bm, &cm);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 36; ++j) {
        float expected = bias[j];
        for (int k = 0; k < 768; ++k)
          expected += a[i][k] * w[j][k];
        if (c[i][j] != expected) {
          printf("linear bias mismatch: run %d row %d column %d: %d != %d\n",
                 run, i, j, (int)c[i][j], (int)expected);
          return 1;
        }
      }
  }
  puts("MXINT8 linear layer with its bias on mxint8mm passed");
  return 0;
}
