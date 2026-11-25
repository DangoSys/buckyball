#include <kernel.h>
#include <stdint.h>
#include <stdio.h>

struct tensor4 {
  float *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
struct tensor3 {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor4 *, struct tensor4 *,
                                 struct tensor3 *);
static float q[1][5][6][64] __attribute__((aligned(64)));
static float k[1][5][2][64] __attribute__((aligned(64)));
static float c[2][15][5] __attribute__((aligned(64)));

int main(void) {
  struct kernel kernel;
  struct kernel_slot slots[4];
  kernel_init(&kernel, slots, 4);
  struct tensor4 qm = {
      &q[0][0][0][0], &q[0][0][0][0], 0, {1, 5, 6, 64}, {1920, 384, 64, 1}};
  struct tensor4 km = {
      &k[0][0][0][0], &k[0][0][0][0], 0, {1, 5, 2, 64}, {640, 128, 64, 1}};
  struct tensor3 cm = {&c[0][0][0], &c[0][0][0], 0, {2, 15, 5}, {75, 5, 1}};
  for (int t = 0; t < 5; ++t)
    for (int d = 0; d < 64; ++d) {
      for (int h = 0; h < 6; ++h)
        q[0][t][h][d] = (float)((t * 3 + h * 5 + d) % 7 - 3);
      for (int g = 0; g < 2; ++g)
        k[0][t][g][d] = (float)((t + g * 2 + d * 3) % 5 - 2);
    }
  _mlir_ciface_compute(&qm, &km, &cm);
  for (int g = 0; g < 2; ++g)
    for (int r = 0; r < 15; ++r)
      for (int j = 0; j < 5; ++j) {
        int h = g * 3 + r / 5, t = r % 5;
        float expected = 0;
        for (int d = 0; d < 64; ++d)
          expected += q[0][t][h][d] * k[0][j][g][d];
        if (c[g][r][j] != expected) {
          printf("GQA scores mismatch: head %d row %d key %d: %d != %d\n", g, r,
                 j, (int)c[g][r][j], (int)expected);
          return 1;
        }
      }
  puts("Compiled grouped-query scores: strided query rows and key tail passed");
  return 0;
}
