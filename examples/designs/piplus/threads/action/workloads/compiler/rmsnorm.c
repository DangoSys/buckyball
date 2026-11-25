#include <kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/* The compiled RMSNorm of twenty rows of 960 on LayerNormBall (rms),
 * checked against the normalization of each row in double precision. */
struct matrix {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct vector {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern void _mlir_ciface_compute(struct matrix *, struct vector *,
                                 struct matrix *);
static float x[20][960] __attribute__((aligned(64)));
static float y[20][960] __attribute__((aligned(64)));
static float g[960] __attribute__((aligned(64)));

int main(void) {
  struct kernel kernel;
  struct kernel_slot slots[4];
  kernel_init(&kernel, slots, 4);
  struct matrix xm = {&x[0][0], &x[0][0], 0, {20, 960}, {960, 1}};
  struct matrix ym = {&y[0][0], &y[0][0], 0, {20, 960}, {960, 1}};
  struct vector gm = {g, g, 0, {960}, {1}};
  for (int k = 0; k < 960; ++k)
    g[k] = 0.75f + (float)(k % 11) / 16.0f;
  for (int i = 0; i < 20; ++i)
    for (int k = 0; k < 960; ++k)
      x[i][k] = (float)((k * 7 + i * 13) % 29 - 14) * 0.25f + (float)i * 3.0f;
  _mlir_ciface_compute(&xm, &gm, &ym);
  for (int i = 0; i < 20; ++i) {
    double square = 0;
    for (int k = 0; k < 960; ++k)
      square += (double)x[i][k] * x[i][k];
    double r = 1 / sqrt(square / 960 + 1e-5);
    for (int k = 0; k < 960; ++k) {
      double expected = x[i][k] * r * g[k];
      if (fabs(y[i][k] - expected) > 1e-4 * (1 + fabs(expected))) {
        printf("rmsnorm mismatch: row %d column %d: %g != %g\n", i, k,
               (double)y[i][k], expected);
        return 1;
      }
    }
  }
  puts("RMSNorm of 960-wide rows compiled to LayerNormBall passed");
  return 0;
}
