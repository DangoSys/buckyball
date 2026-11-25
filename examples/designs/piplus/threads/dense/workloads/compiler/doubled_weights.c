#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct blocks {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct matrix *, struct blocks *,
                                 struct blocks *, struct matrix *,
                                 struct matrix *);
// Two weights [120, 768] in the layout of the mxint8mm Ball: 1 block(s) of
// 768 values, a row of a block the codes, the exponent bytes, zero bytes.
static float a[37][768], bias[120], output[37][120];
static int8_t first[1][120][800], second[1][120][800];
int main(void) {
  struct kernel kernel;
  struct kernel_slot slots[16];
  kernel_init(&kernel, slots, 16);
  struct matrix am = {a, a, 0, {37, 768}, {768, 1}};
  struct blocks w1 = {first, first, 0, {1, 120, 800}, {120 * 800, 800, 1}};
  struct blocks w2 = {second, second, 0, {1, 120, 800}, {120 * 800, 800, 1}};
  struct matrix bm = {bias, bias, 0, {1, 120}, {120, 1}};
  struct matrix cm = {output, output, 0, {37, 120}, {120, 1}};
  for (int i = 0; i < 37; ++i)
    for (int k = 0; k < 768; ++k)
      a[i][k] = (i + k) % 5 - 2;
  for (int j = 0; j < 120; ++j) {
    bias[j] = (float)(j % 9 - 4);
    for (int k = 0; k < 768; ++k) {
      first[k / 768][j][k % 768] = (j + k) % 7 - 3;
      second[k / 768][j][k % 768] = (j * 3 + k) % 5 - 2;
      first[k / 768][j][768 + k % 768 / 32] = 127;
      second[k / 768][j][768 + k % 768 / 32] = 121; // 2^-6
    }
  }
  _mlir_ciface_compute(&am, &w1, &w2, &bm, &cm);
  for (int i = 0; i < 37; ++i)
    for (int j = 0; j < 120; ++j) {
      float expected = bias[j];
      for (int k = 0; k < 768; ++k)
        expected += a[i][k] * ((j + k) % 7 - 3) +
                    a[i][k] * ((j * 3 + k) % 5 - 2) / 64.0f;
      if (output[i][j] != expected) {
        printf("doubled weights mismatch at %d,%d: %d != %d (x64)\n", i, j,
               (int)(output[i][j] * 64), (int)(expected * 64));
        return 1;
      }
    }
  puts("Two MXINT8 weight terms with a bias on mxint8mm: weights passed");
  return 0;
}
