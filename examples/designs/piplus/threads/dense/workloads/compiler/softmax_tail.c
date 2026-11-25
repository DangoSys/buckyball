#include <kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/* The compiled softmax of twenty rows of 241 on SoftmaxBall, checked against
 * the softmax of each row in double precision: every value within 1e-6. */
struct matrix {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *);
static float x[20][241] __attribute__((aligned(64)));
static float y[20][241] __attribute__((aligned(64)));

int main(void) {
  struct kernel kernel;
  struct kernel_slot slots[16];
  kernel_init(&kernel, slots, 16);
  struct matrix xm = {&x[0][0], &x[0][0], 0, {20, 241}, {241, 1}};
  struct matrix ym = {&y[0][0], &y[0][0], 0, {20, 241}, {241, 1}};
  for (int i = 0; i < 20; ++i)
    for (int k = 0; k < 241; ++k)
      x[i][k] = (float)((k * 7 + i * 13) % 29 - 14) * 0.25f * (float)(i + 1);
  _mlir_ciface_compute(&xm, &ym);
  for (int i = 0; i < 20; ++i) {
    double m = x[i][0], s = 0;
    for (int k = 1; k < 241; ++k)
      m = x[i][k] > m ? x[i][k] : m;
    for (int k = 0; k < 241; ++k)
      s += exp(x[i][k] - m);
    for (int k = 0; k < 241; ++k) {
      double expected = exp(x[i][k] - m) / s;
      if (fabs(y[i][k] - expected) > 1e-6) {
        printf("softmax mismatch: row %d column %d: %g != %g\n", i, k,
               (double)y[i][k], expected);
        return 1;
      }
    }
  }
  puts("softmax of 241-wide rows compiled to SoftmaxBall passed");
  return 0;
}
