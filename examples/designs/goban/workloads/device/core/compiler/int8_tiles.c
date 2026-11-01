#include <engine.h>
#include <stdint.h>
#include <stdio.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct layout {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct matrix *, struct layout *,
                                 struct matrix *, struct matrix *,
                                 struct matrix *, struct matrix *);
static float a[3][273], output[3][19];
static int8_t w[19][273];
// w in the smatmul B layout: slice s, line l, group g, channel c.
static int8_t packed[2][32][9][16];
static float as[3] = {0.25f, 0.5f, 1.0f}, mult[3] = {4, 2, 1}, ws[19];
int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  struct matrix am = {a, a, 0, {3, 273}, {273, 1}};
  struct layout wm = {packed, packed, 0, {2, 32, 144}, {4608, 144, 1}};
  struct matrix mm = {mult, mult, 0, {3, 1}, {1, 1}};
  struct matrix asm_ = {as, as, 0, {3, 1}, {1, 1}};
  struct matrix wsm = {ws, ws, 0, {19, 1}, {1, 1}};
  struct matrix cm = {output, output, 0, {3, 19}, {19, 1}};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 273; ++k)
      a[i][k] = ((i + k) % 7 - 3) * as[i];
  for (int j = 0; j < 19; ++j) {
    ws[j] = j % 2 ? 0.5f : 1.0f;
    for (int k = 0; k < 273; ++k)
      w[j][k] = (j * 3 + k) % 5 - 2;
  }
  for (int j = 0; j < 19; ++j)
    for (int k = 0; k < 273; ++k)
      packed[j / 16][k % 32][k / 32][j % 16] = w[j][k];
  _mlir_ciface_compute(&am, &wm, &mm, &asm_, &wsm, &cm);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 19; ++j) {
      float expected = 0;
      for (int k = 0; k < 273; ++k)
        expected += a[i][k] * w[j][k] * ws[j];
      if (output[i][j] != expected) {
        printf("INT8 tile mismatch at %d,%d: scaled values %d != %d\n", i, j,
               (int)(output[i][j] * 8), (int)(expected * 8));
        return 1;
      }
    }
  puts("Compiled INT8: row/channel scales, K accumulation and output tails "
       "passed");
  return 0;
}
