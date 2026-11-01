#include <engine.h>
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
                                 struct matrix *);
// A weight [120, 768] in the layout of the mxint8mm Ball: 6 blocks of 128
// values, a row of a block the 128 codes, 4 exponent bytes, 12 zero bytes.
static float a[37][768], output[37][120];
static int8_t weights[6][120][144];
int main(void) {
  struct engine engine;
  struct engine_slot slots[1];
  engine_init(&engine, slots, 1);
  struct matrix am = {a, a, 0, {37, 768}, {768, 1}};
  struct blocks wm = {weights, weights, 0, {6, 120, 144}, {120 * 144, 144, 1}};
  struct matrix cm = {output, output, 0, {37, 120}, {120, 1}};
  for (int i = 0; i < 37; ++i)
    for (int k = 0; k < 768; ++k)
      a[i][k] = (i + k) % 5 - 2;
  for (int j = 0; j < 120; ++j)
    for (int k = 0; k < 768; ++k) {
      weights[k / 128][j][k % 128] = (j + k) % 7 - 3;
      weights[k / 128][j][128 + k % 128 / 32] = (int8_t)(126 + (k / 32) % 3);
    }
  _mlir_ciface_compute(&am, &wm, &cm);
  for (int i = 0; i < 37; ++i)
    for (int j = 0; j < 120; ++j) {
      float expected = 0;
      for (int k = 0; k < 768; ++k) {
        int exponent = (k / 32) % 3;
        float scale = exponent == 0 ? 0.5f : exponent == 1 ? 1.0f : 2.0f;
        expected += a[i][k] * ((j + k) % 7 - 3) * scale;
      }
      if (output[i][j] != expected) {
        printf("Packed weight mismatch at %d,%d: doubled values %d != %d\n", i,
               j, (int)(output[i][j] * 2), (int)(expected * 2));
        return 1;
      }
    }
  puts("Compiled Ball-layout weights: M and N tiles, padded rows passed");
  return 0;
}
