#include <kernel.h>
#include <stdint.h>
#include <stdio.h>

/* An attention of 2 heads, 20 queries and 32 keys of 64 values. Query i of
 * head h has 16 at value i and key (3 i + h) % 32 has 64 there, so its score
 * is 1024 at that key and 0 at the others: scaled by 1/8, the softmax is
 * exactly 1 there and 0 elsewhere, and the output row is that key's row of
 * v, small integers MXINT8 holds exactly. */
struct tensor {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *,
                                 struct tensor *, struct tensor *);
static float q[2][20][64] __attribute__((aligned(64)));
static float k[2][32][64] __attribute__((aligned(64)));
static float v[2][32][64] __attribute__((aligned(64)));
static float o[2][20][64] __attribute__((aligned(64)));

int main(void) {
  struct kernel kernel;
  struct kernel_slot slots[16];
  kernel_init(&kernel, slots, 16);
  struct tensor qm = {&q[0][0][0], &q[0][0][0], 0, {2, 20, 64}, {1280, 64, 1}};
  struct tensor km = {&k[0][0][0], &k[0][0][0], 0, {2, 32, 64}, {2048, 64, 1}};
  struct tensor vm = {&v[0][0][0], &v[0][0][0], 0, {2, 32, 64}, {2048, 64, 1}};
  struct tensor om = {&o[0][0][0], &o[0][0][0], 0, {2, 20, 64}, {1280, 64, 1}};
  for (int h = 0; h < 2; ++h) {
    for (int i = 0; i < 20; ++i) {
      q[h][i][i] = 16;
      k[h][(3 * i + h) % 32][i] = 64;
    }
    for (int t = 0; t < 32; ++t)
      for (int d = 0; d < 64; ++d)
        v[h][t][d] = (float)((t * 7 + d + h) % 13 - 6);
  }
  _mlir_ciface_compute(&qm, &km, &vm, &om);
  for (int h = 0; h < 2; ++h)
    for (int i = 0; i < 20; ++i)
      for (int d = 0; d < 64; ++d) {
        float expected = v[h][(3 * i + h) % 32][d];
        if (o[h][i][d] != expected) {
          printf("Attention mismatch: head %d query %d value %d: %d != %d\n", h,
                 i, d, (int)o[h][i][d], (int)expected);
          return 1;
        }
      }
  puts("Compiled attention in the thread's banks passed");
  return 0;
}
