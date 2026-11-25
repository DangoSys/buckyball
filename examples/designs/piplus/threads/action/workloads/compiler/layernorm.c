#include <kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/* The compiled layer normalization of twenty rows of 768 on LayerNormBall,
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
                                 struct vector *, struct matrix *);
static float x[20][768] __attribute__((aligned(64)));
static float y[20][768] __attribute__((aligned(64)));
static float g[768] __attribute__((aligned(64)));
static float bt[768] __attribute__((aligned(64)));

int main(void) {
  struct kernel kernel;
  struct kernel_slot slots[4];
  kernel_init(&kernel, slots, 4);
  struct matrix xm = {&x[0][0], &x[0][0], 0, {20, 768}, {768, 1}};
  struct matrix ym = {&y[0][0], &y[0][0], 0, {20, 768}, {768, 1}};
  struct vector gm = {g, g, 0, {768}, {1}};
  struct vector bm = {bt, bt, 0, {768}, {1}};
  for (int k = 0; k < 768; ++k) {
    g[k] = 0.75f + (float)(k % 11) / 16.0f;
    bt[k] = (float)(k % 5) - 2.0f;
  }
  for (int i = 0; i < 20; ++i)
    for (int k = 0; k < 768; ++k)
      x[i][k] = (float)((k * 7 + i * 13) % 29 - 14) * 0.25f + (float)i * 3.0f;
  _mlir_ciface_compute(&xm, &gm, &bm, &ym);
  for (int i = 0; i < 20; ++i) {
    double mean = 0, var = 0;
    for (int k = 0; k < 768; ++k)
      mean += x[i][k];
    mean /= 768;
    for (int k = 0; k < 768; ++k)
      var += (x[i][k] - mean) * (x[i][k] - mean);
    var /= 768;
    double r = 1 / sqrt(var + 1e-6);
    for (int k = 0; k < 768; ++k) {
      double expected = (x[i][k] - mean) * r * g[k] + bt[k];
      if (fabs(y[i][k] - expected) > 1e-4 * (1 + fabs(expected))) {
        printf("layernorm mismatch: row %d column %d: %g != %g\n", i, k,
               (double)y[i][k], expected);
        return 1;
      }
    }
  }
  puts("layernorm of 768-wide rows compiled to LayerNormBall passed");
  return 0;
}
