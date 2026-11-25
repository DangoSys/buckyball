#include <initializer_list>
#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <testutils.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct layout {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" void _mlir_ciface_compute(struct matrix *, struct layout *,
                                     struct matrix *, struct matrix *,
                                     struct matrix *, struct matrix *);
extern "C" void _mlir_ciface_compute_one(struct matrix *, struct layout *,
                                         struct matrix *, struct matrix *,
                                         struct matrix *, struct matrix *);
static float output[3][19];
struct inputs {
  alignas(64) float a[3][273];
  float as[3], mult[3], ws[19], expected[3][19];
  int8_t w[19][273];
  alignas(64) int8_t packed[2][32][9][16];
};
alignas(64) static constexpr inputs data = [] {
  inputs result{};
  for (int i = 0; i < 3; ++i) {
    result.as[i] = i == 0 ? 0.25f : i == 1 ? 0.5f : 1.0f;
    result.mult[i] = i == 0 ? 4 : i == 1 ? 2 : 1;
    for (int k = 0; k < 273; ++k)
      result.a[i][k] = ((i + k) % 7 - 3) * result.as[i];
  }
  for (int j = 0; j < 19; ++j) {
    result.ws[j] = j % 2 ? 0.5f : 1.0f;
    for (int k = 0; k < 273; ++k) {
      result.w[j][k] = (j * 3 + k) % 5 - 2;
      result.packed[j / 16][k % 32][k / 32][j % 16] = result.w[j][k];
    }
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 19; ++j)
      for (int k = 0; k < 273; ++k)
        result.expected[i][j] += result.a[i][k] * result.w[j][k] * result.ws[j];
  return result;
}();
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  struct matrix am = {(void *)data.a, (void *)data.a, 0, {3, 273}, {273, 1}};
  struct layout wm = {(void *)data.packed,
                      (void *)data.packed,
                      0,
                      {2, 32, 144},
                      {4608, 144, 1}};
  struct matrix mm = {(void *)data.mult, (void *)data.mult, 0, {3, 1}, {1, 1}};
  struct matrix asm_ = {(void *)data.as, (void *)data.as, 0, {3, 1}, {1, 1}};
  struct matrix wsm = {(void *)data.ws, (void *)data.ws, 0, {19, 1}, {1, 1}};
  struct matrix cm = {output, output, 0, {3, 19}, {19, 1}};
  for (int rows : {3, 1}) {
    am.sizes[0] = mm.sizes[0] = asm_.sizes[0] = cm.sizes[0] = rows;
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < 19; ++j)
        output[i][j] = 12345.0f;
    uint64_t start = read_counter();
    (rows == 1 ? _mlir_ciface_compute_one
               : _mlir_ciface_compute)(&am, &wm, &mm, &asm_, &wsm, &cm);
    printf("Operator rows %d cycles: %lu\n", rows,
           (unsigned long)(read_counter() - start));
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < 19; ++j) {
        float expected = data.expected[i][j];
        if (output[i][j] != expected) {
          printf("INT8 tile mismatch at %d,%d: scaled values %d != %d\n", i, j,
                 (int)(output[i][j] * 8), (int)(expected * 8));
          return 1;
        }
      }
  }
  puts("Compiled INT8: row/channel scales, K accumulation and output tails "
       "passed");
  return 0;
}
